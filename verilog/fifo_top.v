`timescale 1ns/1ps
///////////////////////////////////////////////////////////////////////////////
// fifo_top.v — Network Processor Integration
//
// Register map matches ids.xml exactly: 6 SW + 5 HW registers.
// The NetFPGA build system generates macros from ids.xml:
//   `FIFO_BLOCK_ADDR       — base address tag for generic_regs
//   `FIFO_REG_ADDR_WIDTH   — address bits (log2(blocksize/4) = log2(64) = 6)
//
// ─────────────────────────────────────────────────────────────────────
// SW registers (host writes):
//   offset 0x00  cmd          [1:0]=mode [2]=fifo_rst [3]=arm_start
//                             [4]=gpu_start [5]=gpu2_start
//   offset 0x04  proc_addr    byte address for BRAM/IMEM/DMEM access
//   offset 0x08  wdata_hi     data[63:32] for FIFO BRAM writes
//   offset 0x0c  wdata_lo     data[31:0]  also used as 32-bit prog word
//   offset 0x10  wdata_ctrl   ctrl[7:0] + host_we trigger
//   offset 0x14  rdata_sel    [0]=bram/gpu2  [1]=arm_prog_en
//                             [2]=gpu_prog_en [3]=arm_dmem_sel
//                             [4]=gpu_dmem_sel [5]=arm_prog_we (edge)
//                             [6]=gpu_prog_we  (edge)
//
// HW registers (host reads):
//   offset 0x18  status       [1:0]=mode [2]=pkt_ready [15:8]=pkt_len
//   offset 0x1c  rdata_hi     FIFO BRAM data[63:32] at proc_addr
//   offset 0x20  rdata_lo     FIFO BRAM data[31:0]
//   offset 0x24  rdata_ctrl   {24'b0, ctrl[7:0]}
//   offset 0x28  pointers     [0]=arm_done [1]=arm_run [2]=gpu_done
//                             [3]=gpu_run  [4]=gpu2_done [5]=gpu2_run
//                             [15:8]=tail  [23:16]=head
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

   // ==================================================================
   // SW Registers (names match ids.xml register names)
   // ==================================================================
   wire [31:0] sw_cmd;
   wire [31:0] sw_proc_addr;
   wire [31:0] sw_wdata_hi;
   wire [31:0] sw_wdata_lo;
   wire [31:0] sw_wdata_ctrl;
   wire [31:0] sw_rdata_sel;

   // ── cmd fields ─────────────────────────────────────────────────
   wire [1:0] fifo_mode = sw_cmd[1:0];
   wire       cmd_reset = sw_cmd[2];

   // ── rdata_sel fields ───────────────────────────────────────────
   wire arm_prog_en  = sw_rdata_sel[1];
   wire gpu_prog_en  = sw_rdata_sel[2];
   wire arm_dmem_sel = sw_rdata_sel[3];
   wire gpu_dmem_sel = sw_rdata_sel[4];

   // ==================================================================
   // Edge detection — start pulses (cmd[3:5]) and prog_we (rdata_sel[5:6])
   // ==================================================================
   reg cmd3_prev, cmd4_prev, cmd5_prev;
   reg rsel5_prev, rsel6_prev;

   wire arm_start  = sw_cmd[3]        & ~cmd3_prev;
   wire gpu_start  = sw_cmd[4]        & ~cmd4_prev;
   wire gpu2_start = sw_cmd[5]        & ~cmd5_prev;
   wire arm_prog_we = sw_rdata_sel[5] & ~rsel5_prev;
   wire gpu_prog_we = sw_rdata_sel[6] & ~rsel6_prev;

   always @(posedge clk) begin
      if (reset) begin
         cmd3_prev  <= 1'b0; cmd4_prev  <= 1'b0; cmd5_prev  <= 1'b0;
         rsel5_prev <= 1'b0; rsel6_prev <= 1'b0;
      end else begin
         cmd3_prev  <= sw_cmd[3];        cmd4_prev  <= sw_cmd[4];
         cmd5_prev  <= sw_cmd[5];
         rsel5_prev <= sw_rdata_sel[5];  rsel6_prev <= sw_rdata_sel[6];
      end
   end

   // ==================================================================
   // ARM CPU — PCI-style programming interface (Lab 6 pattern)
   // Prog data path: proc_addr = byte addr, wdata_lo = 32-bit word
   // ==================================================================
   wire        arm_done;
   wire        arm_running;
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

   // ==================================================================
   // GPU — BFloat16 tensor core, PCI-style programming interface
   // ==================================================================
   wire        gpu_done;
   wire        gpu_running;
   wire [71:0] proc_rdata;   // shared FIFO BRAM Port B output

   wire [7:0]  gpu_bram_addr;
   wire [71:0] gpu_bram_wdata;
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
      .bram_we        (gpu_bram_we)
   );

   // ==================================================================
   // GPU2 / Simple CPU placeholder (ids.xml bit [4:5] of pointers)
   // Tie off until GPU2 is connected.
   // ==================================================================
   wire gpu2_done    = 1'b0;
   wire gpu2_running = 1'b0;
   // gpu2_start decoded above; connect to GPU2 when ready.

   // ==================================================================
   // Host FIFO BRAM write trigger — edge on sw_wdata_ctrl
   // Active only in PROC mode and when no processor is running.
   // ==================================================================
   reg [31:0] sw_wdata_ctrl_prev;
   wire host_we = (sw_wdata_ctrl != sw_wdata_ctrl_prev)
               && (fifo_mode == 2'b01)
               && !arm_running && !gpu_running;

   always @(posedge clk) begin
      if (reset) sw_wdata_ctrl_prev <= 32'b0;
      else       sw_wdata_ctrl_prev <= sw_wdata_ctrl;
   end

   // ==================================================================
   // GPU BRAM programming path (gpu_prog_we & gpu_dmem_sel=1)
   // Allows 72-bit pre-loading of GPU operands into shared BRAM
   // via proc_addr + wdata_hi + wdata_lo + wdata_ctrl.
   // ==================================================================
   wire        gpu_bram_prog_we   = gpu_prog_we & gpu_dmem_sel;
   wire [7:0]  gpu_bram_prog_addr = sw_proc_addr[9:2];
   wire [71:0] gpu_bram_prog_data = {sw_wdata_ctrl[7:0],
                                      sw_wdata_hi,
                                      sw_wdata_lo};

   // ==================================================================
   // BRAM Port B MUX
   // Priority: GPU runtime > GPU BRAM prog > host PROC write
   // ==================================================================
   wire [7:0]  mux_addr;
   wire [71:0] mux_wdata;
   wire        mux_we;

   assign mux_addr  = gpu_running      ? gpu_bram_addr       :
                      gpu_bram_prog_we ? gpu_bram_prog_addr   :
                                         sw_proc_addr[7:0];

   assign mux_wdata = gpu_running      ? gpu_bram_wdata       :
                      gpu_bram_prog_we ? gpu_bram_prog_data    :
                      {sw_wdata_ctrl[7:0], sw_wdata_hi, sw_wdata_lo};

   assign mux_we    = gpu_running      ? gpu_bram_we          :
                      gpu_bram_prog_we ? 1'b1                  :
                                         host_we;

   // ==================================================================
   // Convertible FIFO instance
   // ==================================================================
   wire        pkt_ready;
   wire [7:0]  pkt_len;
   wire [7:0]  head_addr, tail_addr;
   wire [63:0] fifo_out_data;
   wire [7:0]  fifo_out_ctrl;
   wire        fifo_out_wr;
   wire        fifo_in_rdy;

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

   // ==================================================================
   // Passthrough MUX
   // ==================================================================
   assign out_data = (fifo_mode == 2'b00) ? in_data     : fifo_out_data;
   assign out_ctrl = (fifo_mode == 2'b00) ? in_ctrl     : fifo_out_ctrl;
   assign out_wr   = (fifo_mode == 2'b00) ? in_wr       : fifo_out_wr;
   assign in_rdy   = (fifo_mode == 2'b00) ? out_rdy     :
                     (fifo_mode == 2'b11) ? fifo_in_rdy : 1'b0;

   // ==================================================================
   // HW Registers (names match ids.xml register names)
   // ==================================================================
   wire [31:0] hw_status    = {16'b0, pkt_len, 5'b0, pkt_ready, fifo_mode};
   wire [31:0] hw_rdata_hi  = proc_rdata[63:32];
   wire [31:0] hw_rdata_lo  = proc_rdata[31:0];
   wire [31:0] hw_rdata_ctrl = {24'b0, proc_rdata[71:64]};
   wire [31:0] hw_pointers  = {8'b0,
                                head_addr,     // [23:16]
                                tail_addr,     // [15:8]
                                2'b0,
                                gpu2_running,  // [5]
                                gpu2_done,     // [4]
                                gpu_running,   // [3]
                                gpu_done,      // [2]
                                arm_running,   // [1]
                                arm_done};     // [0]

   // ==================================================================
   // generic_regs: 6 SW + 5 HW  (matches ids.xml exactly)
   //
   // generic_regs packs the arrays with reg0 in the LOWEST bits:
   //   software_regs[31:0]    = sw_cmd       (reg 0, offset 0x00)
   //   software_regs[63:32]   = sw_proc_addr (reg 1, offset 0x04)
   //   software_regs[95:64]   = sw_wdata_hi  (reg 2, offset 0x08)
   //   software_regs[127:96]  = sw_wdata_lo  (reg 3, offset 0x0c)
   //   software_regs[159:128] = sw_wdata_ctrl(reg 4, offset 0x10)
   //   software_regs[191:160] = sw_rdata_sel (reg 5, offset 0x14)
   //
   //   hardware_regs[31:0]    = hw_status    (reg 0, offset 0x18)
   //   hardware_regs[63:32]   = hw_rdata_hi  (reg 1, offset 0x1c)
   //   hardware_regs[95:64]   = hw_rdata_lo  (reg 2, offset 0x20)
   //   hardware_regs[127:96]  = hw_rdata_ctrl(reg 3, offset 0x24)
   //   hardware_regs[159:128] = hw_pointers  (reg 4, offset 0x28)
   // ==================================================================
   generic_regs
   #(
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
      
      // Software interface
      .software_regs ({sw_rdata_sel,
                       sw_wdata_ctrl,
                       sw_wdata_lo,
                       sw_wdata_hi,
                       sw_proc_addr,
                       sw_cmd}),

      // Hardware interface
      .hardware_regs ({hw_pointers,
                       hw_rdata_ctrl,
                       hw_rdata_lo,
                       hw_rdata_hi,
                       hw_status}),
      .clk   (clk),
      .reset (reset)
   );

endmodule
