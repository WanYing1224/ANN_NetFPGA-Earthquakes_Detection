`timescale 1ns/1ps
///////////////////////////////////////////////////////////////////////////////
// fifo_top.v — Network Processor Integration (Triple BRAM)
//
// Register map matches ids.xml exactly: 6 SW + 5 HW registers.
//
// Memory Map (host access via sw_proc_addr):
//   Bits [11:10] = 2'b00  →  Feature BRAM (convertible_fifo)
//   Bits [11:10] = 2'b01  →  Weight BRAM
//   Bits [11:10] = 2'b11  →  Bias ROM (read-only, initialized via initial block)
///////////////////////////////////////////////////////////////////////////////

module fifo_top
   #(
      parameter DATA_WIDTH = 64,
      parameter CTRL_WIDTH = DATA_WIDTH/8,
      parameter UDP_REG_SRC_WIDTH = 2
   )
   (
      input  [DATA_WIDTH-1:0]             in_data,
      input  [CTRL_WIDTH-1:0]             in_ctrl,
      input                               in_wr,
      output                              in_rdy,
      output [DATA_WIDTH-1:0]             out_data,
      output [CTRL_WIDTH-1:0]             out_ctrl,
      output                              out_wr,
      input                               out_rdy,
      input                               reg_req_in,
      input                               reg_ack_in,
      input                               reg_rd_wr_L_in,
      input  [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_in,
      input  [`CPCI_NF2_DATA_WIDTH-1:0]   reg_data_in,
      input  [UDP_REG_SRC_WIDTH-1:0]      reg_src_in,
      output                              reg_req_out,
      output                              reg_ack_out,
      output                              reg_rd_wr_L_out,
      output [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_out,
      output [`CPCI_NF2_DATA_WIDTH-1:0]   reg_data_out,
      output [UDP_REG_SRC_WIDTH-1:0]      reg_src_out,
      input                               reset,
      input                               clk
   );

   // SW Registers (names match ids.xml register names)
   wire [31:0] sw_cmd, sw_proc_addr, sw_wdata_hi, sw_wdata_lo, sw_wdata_ctrl, sw_rdata_sel;

   // cmd fields
   wire [1:0] fifo_mode = sw_cmd[1:0];
   wire       cmd_reset = sw_cmd[2];

   // rdata_sel fields
   wire arm_prog_en  = sw_rdata_sel[1];
   wire gpu_prog_en  = sw_rdata_sel[2];
   wire arm_dmem_sel = sw_rdata_sel[3];
   wire gpu_dmem_sel = sw_rdata_sel[4];

   // Edge detection — start pulses
   reg cmd3_prev, cmd4_prev, cmd5_prev, rsel5_prev, rsel6_prev;
   wire arm_start   = sw_cmd[3]       & ~cmd3_prev;
   wire gpu_start   = sw_cmd[4]       & ~cmd4_prev;
   wire gpu2_start  = sw_cmd[5]       & ~cmd5_prev;
   wire arm_prog_we = sw_rdata_sel[5] & ~rsel5_prev;
   wire gpu_prog_we = sw_rdata_sel[6] & ~rsel6_prev;

   always @(posedge clk) begin
      if (reset) begin
         cmd3_prev <= 1'b0; cmd4_prev <= 1'b0; cmd5_prev <= 1'b0;
         rsel5_prev <= 1'b0; rsel6_prev <= 1'b0;
      end else begin
         cmd3_prev <= sw_cmd[3];       cmd4_prev <= sw_cmd[4];       cmd5_prev <= sw_cmd[5];
         rsel5_prev <= sw_rdata_sel[5]; rsel6_prev <= sw_rdata_sel[6];
      end
   end

   // ─── ARM CPU ──────────────────────────────────────────────────────────
   wire        arm_done, arm_running;
   wire [31:0] arm_dmem_rdata;
   arm_cpu_wrapper arm_inst (
      .clk         (clk),
      .rst_n       (~reset),
      .prog_en     (arm_prog_en),
      .prog_we     (arm_prog_we),
      .dmem_sel    (arm_dmem_sel),
      .prog_addr   (sw_proc_addr),
      .prog_wdata  (sw_wdata_lo),
      .prog_rdata  (arm_dmem_rdata),
      .cpu_done    (arm_done),
      .cpu_running (arm_running)
   );

   // ─── GPU TENSOR CORE ──────────────────────────────────────────────────
   wire        gpu_done, gpu_running;
   wire [7:0]  gpu_bram_addr, gpu_weight_addr, gpu_bias_addr;
   wire [71:0] gpu_bram_wdata;
   wire [71:0] proc_rdata, weight_rdata_out, bias_rdata_out;
   wire        gpu_bram_we;

   gpu_net gpu_inst (
      .clk            (clk),
      .rst_n          (~reset),
      .start          (gpu_start),
      .done           (gpu_done),
      .running        (gpu_running),
      .gpu_prog_en    (gpu_prog_en),
      .gpu_prog_we    (gpu_prog_we),
      .gpu_dmem_sel   (gpu_dmem_sel),
      .gpu_prog_addr  (sw_proc_addr),
      .gpu_prog_wdata (sw_wdata_lo),

      .bram_addr      (gpu_bram_addr),
      .bram_wdata     (gpu_bram_wdata),
      .bram_rdata     (proc_rdata),
      .bram_we        (gpu_bram_we),

      .weight_addr    (gpu_weight_addr),
      .weight_rdata   (weight_rdata_out),

      .bias_addr      (gpu_bias_addr),
      .bias_rdata     (bias_rdata_out)
   );

   wire gpu2_done    = 1'b0;
   wire gpu2_running = 1'b0;

   // ─── HOST ADDRESS DECODER & WRITE LOGIC ───────────────────────────────
   reg [31:0] sw_wdata_ctrl_prev;
   always @(posedge clk) begin
      if (reset) sw_wdata_ctrl_prev <= 32'b0;
      else       sw_wdata_ctrl_prev <= sw_wdata_ctrl;
   end

   // 3-zone address decoder using bits [11:10]:
   //   2'b00 → Feature BRAM
   //   2'b01 → Weight BRAM
   //   2'b11 → Bias ROM (read-only)
   wire [1:0] mem_zone       = sw_proc_addr[11:10];
   wire       sel_feature    = (mem_zone == 2'b00);
   wire       sel_weight     = (mem_zone == 2'b01);
   wire       sel_bias       = (mem_zone == 2'b11);

   // Host writes (PROC mode, no CPU/GPU running)
   wire host_we = (sw_wdata_ctrl != sw_wdata_ctrl_prev) && (fifo_mode == 2'b01) && !arm_running && !gpu_running;

   // GPU pre-load programming path
   wire gpu_bram_prog_we   = gpu_prog_we & gpu_dmem_sel;
   wire [7:0]  gpu_bram_prog_addr = sw_proc_addr[9:2];
   wire [71:0] host_wdata = {sw_wdata_ctrl[7:0], sw_wdata_hi, sw_wdata_lo};

   // Write enables split by address decoder
   // Note: Bias ROM is read-only, so no bias write enable
   wire prog_we_feature = gpu_bram_prog_we & sel_feature;
   wire prog_we_weight  = gpu_bram_prog_we & sel_weight;
   wire host_we_feature = host_we          & sel_feature;
   wire host_we_weight  = host_we          & sel_weight;

   // ─── FEATURE BRAM MUX (convertible_fifo) ──────────────────────────────
   wire [7:0]  mux_addr  = gpu_running ? gpu_bram_addr  : (gpu_bram_prog_we ? gpu_bram_prog_addr : sw_proc_addr[9:2]);
   wire [71:0] mux_wdata = gpu_running ? gpu_bram_wdata : host_wdata;
   wire        mux_we    = gpu_running ? gpu_bram_we    : (gpu_bram_prog_we ? prog_we_feature    : host_we_feature);

   wire        pkt_ready, fifo_out_wr, fifo_in_rdy;
   wire [7:0]  pkt_len, head_addr, tail_addr, fifo_out_ctrl;
   wire [63:0] fifo_out_data;

   convertible_fifo fifo_inst (
      .clk          (clk),
      .reset        (reset),
      .net_in_data  (in_data),
      .net_in_ctrl  (in_ctrl),
      .net_in_wr    (in_wr && (fifo_mode == 2'b11)),
      .net_in_rdy   (fifo_in_rdy),
      .net_out_data (fifo_out_data),
      .net_out_ctrl (fifo_out_ctrl),
      .net_out_wr   (fifo_out_wr),
      .net_out_rdy  (out_rdy),

      .proc_addr    (mux_addr),
      .proc_wdata   (mux_wdata),
      .proc_rdata   (proc_rdata),
      .proc_we      (mux_we),

      .mode         (fifo_mode),
      .cmd_send     (1'b0),
      .cmd_reset    (cmd_reset),
      .pkt_ready    (pkt_ready),
      .pkt_len      (pkt_len),
      .head_addr    (head_addr),
      .tail_addr    (tail_addr)
   );

   // ─── WEIGHT BRAM MUX & INSTANCE ───────────────────────────────────────
   wire [7:0] weight_mux_addr = gpu_running ? gpu_weight_addr : (gpu_bram_prog_we ? gpu_bram_prog_addr : sw_proc_addr[9:2]);
   // GPU only READS from the weight BRAM, it never writes.
   wire weight_mux_we = gpu_running ? 1'b0 : (gpu_bram_prog_we ? prog_we_weight : host_we_weight);
   wire [71:0] weight_rdata_b;

   weight_bram w_bram (
      .clk     (clk),
      // Port A (GPU Runtime)
      .addr_a  (gpu_weight_addr),
      .rdata_a (weight_rdata_out),
      // Port B (Host / GPU Prog MUX)
      .we_b    (weight_mux_we),
      .addr_b  (weight_mux_addr),
      .wdata_b (host_wdata),
      .rdata_b (weight_rdata_b)
   );

   // ─── BIAS ROM INSTANCE (NEW) ──────────────────────────────────────────
   // Bias values are hardcoded in bias_bram.v via initial block.
   // No host write path — host can only read back for verification.
   // Port A serves GPU at runtime; Port B serves host readback.
   wire [71:0] bias_rdata_b;

   bias_bram b_rom (
      .clk     (clk),
      // Port A (GPU Runtime)
      .addr_a  (gpu_bias_addr),
      .dout_a  (bias_rdata_out),
      // Port B (Host readback via PCI register interface)
      .addr_b  (sw_proc_addr[9:2]),
      .dout_b  (bias_rdata_b)
   );

   // ─── READBACK ROUTING & STATUS ────────────────────────────────────────
   // Host readback MUX: route based on address zone [11:10]
   reg [71:0] active_host_rdata;
   always @(*) begin
      case (mem_zone)
         2'b00:   active_host_rdata = proc_rdata;      // Feature BRAM
         2'b01:   active_host_rdata = weight_rdata_b;  // Weight BRAM
         2'b11:   active_host_rdata = bias_rdata_b;    // Bias ROM
         default: active_host_rdata = 72'h0;
      endcase
   end

   assign out_data = (fifo_mode == 2'b00) ? in_data : fifo_out_data;
   assign out_ctrl = (fifo_mode == 2'b00) ? in_ctrl : fifo_out_ctrl;
   assign out_wr   = (fifo_mode == 2'b00) ? in_wr   : fifo_out_wr;
   assign in_rdy   = (fifo_mode == 2'b00) ? out_rdy : (fifo_mode == 2'b11) ? fifo_in_rdy : 1'b0;

   wire [31:0] hw_status     = {16'b0, pkt_len, 5'b0, pkt_ready, fifo_mode};
   wire [31:0] hw_rdata_hi   = active_host_rdata[63:32];
   wire [31:0] hw_rdata_lo   = active_host_rdata[31:0];
   wire [31:0] hw_rdata_ctrl = {24'b0, active_host_rdata[71:64]};
   wire [31:0] hw_pointers   = {8'b0, head_addr, tail_addr, 2'b0, gpu2_running, gpu2_done, gpu_running, gpu_done, arm_running, arm_done};

   // ─── REGISTER SYSTEM (ids.xml) ────────────────────────────────────────
   generic_regs #(
      .UDP_REG_SRC_WIDTH   (UDP_REG_SRC_WIDTH),
      .TAG                 (`FIFO_BLOCK_ADDR),
      .REG_ADDR_WIDTH      (`FIFO_REG_ADDR_WIDTH),
      .NUM_COUNTERS        (0),
      .NUM_SOFTWARE_REGS   (6),
      .NUM_HARDWARE_REGS   (5)
   ) fifo_regs (
      .reg_req_in       (reg_req_in),
      .reg_ack_in       (reg_ack_in),
      .reg_rd_wr_L_in   (reg_rd_wr_L_in),
      .reg_addr_in      (reg_addr_in),
      .reg_data_in      (reg_data_in),
      .reg_src_in       (reg_src_in),
      .reg_req_out      (reg_req_out),
      .reg_ack_out      (reg_ack_out),
      .reg_rd_wr_L_out  (reg_rd_wr_L_out),
      .reg_addr_out     (reg_addr_out),
      .reg_data_out     (reg_data_out),
      .reg_src_out      (reg_src_out),
      .counter_updates  (),
      .counter_decrement(),
      .software_regs ({sw_rdata_sel, sw_wdata_ctrl, sw_wdata_lo, sw_wdata_hi, sw_proc_addr, sw_cmd}),
      .hardware_regs ({hw_pointers, hw_rdata_ctrl, hw_rdata_lo, hw_rdata_hi, hw_status}),
      .clk   (clk),
      .reset (reset)
   );

endmodule
