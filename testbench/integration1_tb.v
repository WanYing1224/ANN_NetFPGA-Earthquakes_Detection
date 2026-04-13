`timescale 1ns/1ps
// ============================================================
// integration_tb.v — ModelSim Testbench
//
// Updated: Dual-BRAM Harvard Architecture for ANN Acceleration.
// Implements Bit 10 Address Decoding for Host BRAM access.
// ============================================================

module integration1_tb;

    // ----------------------------------------------------------------
    // Clock & Reset
    // ----------------------------------------------------------------
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg rst_n;

    // ----------------------------------------------------------------
    // ARM CPU wrapper signals
    // ----------------------------------------------------------------
    reg         arm_prog_en;
    reg         arm_prog_we;
    reg         arm_dmem_sel;
    reg  [31:0] arm_prog_addr;
    reg  [31:0] arm_prog_wdata;
    wire [31:0] arm_prog_rdata;
    wire        arm_cpu_done;
    wire        arm_cpu_running;

    arm_cpu_wrapper arm_dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .prog_en     (arm_prog_en),
        .prog_we     (arm_prog_we),
        .dmem_sel    (arm_dmem_sel),
        .prog_addr   (arm_prog_addr),
        .prog_wdata  (arm_prog_wdata),
        .prog_rdata  (arm_prog_rdata),
        .cpu_done    (arm_cpu_done),
        .cpu_running (arm_cpu_running)
    );

    // ----------------------------------------------------------------
    // GPU Tensor Core signals
    // ----------------------------------------------------------------
    reg         gpu_start_r;
    reg         gpu_prog_en;
    reg         gpu_prog_we;
    reg         gpu_dmem_sel;
    reg  [31:0] gpu_prog_addr;
    reg  [31:0] gpu_prog_wdata;
    wire        gpu_done;
    wire        gpu_running;

    wire [7:0]  gpu_bram_addr;
    wire [71:0] gpu_bram_wdata;
    wire        gpu_bram_we;
    wire [71:0] gpu_bram_rdata;

    wire [7:0]  gpu_weight_addr;
    wire [71:0] gpu_weight_rdata;

    gpu_net gpu_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (gpu_start_r),
        .done           (gpu_done),
        .running        (gpu_running),
        .gpu_prog_en    (gpu_prog_en),
        .gpu_prog_we    (gpu_prog_we),
        .gpu_dmem_sel   (gpu_dmem_sel),
        .gpu_prog_addr  (gpu_prog_addr),
        .gpu_prog_wdata (gpu_prog_wdata),
        
        // Feature BRAM Port
        .bram_addr      (gpu_bram_addr),
        .bram_wdata     (gpu_bram_wdata),
        .bram_rdata     (gpu_bram_rdata),
        .bram_we        (gpu_bram_we),
        
        // Weight BRAM Port
        .weight_addr    (gpu_weight_addr),
        .weight_rdata   (gpu_weight_rdata)
    );

    // ----------------------------------------------------------------
    // Host Address Decoder & Dual-BRAM MUX
    // ----------------------------------------------------------------
    reg  [31:0] host_byte_addr;
    reg  [71:0] host_wdata;
    reg         host_we;

    // Address Decoder: Bit 10 determines target BRAM
    wire sel_weight = host_byte_addr[10];

    wire host_we_feature = host_we & ~sel_weight;
    wire host_we_weight  = host_we &  sel_weight;

    // Feature BRAM MUX
    wire [7:0]  mux_addr_feature  = gpu_running ? gpu_bram_addr  : host_byte_addr[9:2];
    wire [71:0] mux_wdata_feature = gpu_running ? gpu_bram_wdata : host_wdata;
    wire        mux_we_feature    = gpu_running ? gpu_bram_we    : host_we_feature;
    wire [71:0] feature_rdata;

    // Weight BRAM MUX
    wire [7:0]  mux_addr_weight   = gpu_running ? gpu_weight_addr : host_byte_addr[9:2];
    wire        mux_we_weight     = gpu_running ? 1'b0            : host_we_weight; // GPU never writes to weights
    wire [71:0] weight_rdata_b;

    // Routing back to GPU and Host
    assign gpu_bram_rdata = feature_rdata;
    wire [71:0] host_rdata = sel_weight ? weight_rdata_b : feature_rdata;

    // ----------------------------------------------------------------
    // Memory Instantiations
    // ----------------------------------------------------------------
    convertible_fifo fifo_inst (
        .clk          (clk),
        .reset        (~rst_n),
        .net_in_data  (64'b0),
        .net_in_ctrl  (8'b0),
        .net_in_wr    (1'b0),
        .net_in_rdy   (),
        .net_out_data (),
        .net_out_ctrl (),
        .net_out_wr   (),
        .net_out_rdy  (1'b1),
        .proc_addr    (mux_addr_feature),
        .proc_wdata   (mux_wdata_feature),
        .proc_rdata   (feature_rdata),
        .proc_we      (mux_we_feature),
        .mode         (2'b01),        // 2'b01 = MODE_PROC
        .cmd_send     (1'b0),
        .cmd_reset    (1'b0),
        .pkt_ready    (),
        .pkt_len      (),
        .head_addr    (),
        .tail_addr    ()
    );

    weight_bram w_bram (
        .clk     (clk),
        .addr_a  (gpu_weight_addr),
        .rdata_a (gpu_weight_rdata),
        .we_b    (mux_we_weight),
        .addr_b  (mux_addr_weight),
        .wdata_b (host_wdata),
        .rdata_b (weight_rdata_b)
    );

    // ----------------------------------------------------------------
    // Pass / fail counters
    // ----------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;
    integer i;
    reg [31:0] dmem_val;
    reg [71:0] bram_val;

    // ================================================================
    // Tasks
    // ================================================================

    task arm_imem_write(input integer word_idx, input [31:0] instr);
    begin
        arm_prog_addr  = word_idx * 4;
        arm_prog_wdata = instr;
        arm_dmem_sel   = 1'b0;
        @(posedge clk); #1;
        arm_prog_we = 1'b1;
        @(posedge clk); #1;
        arm_prog_we = 1'b0;
    end
    endtask

    task arm_dmem_write(input integer word_idx, input [31:0] data);
    begin
        arm_prog_addr  = word_idx * 4;
        arm_prog_wdata = data;
        arm_dmem_sel   = 1'b1;
        @(posedge clk); #1;
        arm_prog_we = 1'b1;
        @(posedge clk); #1;
        arm_prog_we = 1'b0;
    end
    endtask

    task gpu_imem_write(input integer word_idx, input [31:0] instr);
    begin
        gpu_prog_addr  = word_idx * 4;
        gpu_prog_wdata = instr;
        gpu_dmem_sel   = 1'b0;
        @(posedge clk); #1;
        gpu_prog_we = 1'b1;
        @(posedge clk); #1;
        gpu_prog_we = 1'b0;
    end
    endtask

    // Updated to accept full 32-bit byte address for decoding
    task bram_write_host(input [31:0] byte_addr, input [63:0] data);
    begin
        host_byte_addr = byte_addr;
        host_wdata     = {8'h00, data};
        host_we        = 1'b1;
        @(posedge clk); #1;
        host_we        = 1'b0;
    end
    endtask

    // Updated to accept full 32-bit byte address for decoding
    task bram_read_host(input [31:0] byte_addr, output [71:0] rdata);
    begin
        host_byte_addr = byte_addr;
        host_we        = 1'b0;
        @(posedge clk); #1; // address presented
        @(posedge clk); #1; // data valid
        rdata = host_rdata;
    end
    endtask

    task check64(input [255:0] lbl, input [63:0] got, input [63:0] expected);
    begin
        if (got === expected) begin
            $display("  PASS  %0s", lbl);
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL  %0s", lbl);
            $display("        expected 0x%016h", expected);
            $display("        got      0x%016h", got);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task wait_gpu_done;
        integer n;
    begin
        n = 0;
        while (!gpu_done && n < 10000) begin
            @(posedge clk); #1;
            n = n + 1;
        end
        if (!gpu_done) $display("  TIMEOUT: gpu_done never asserted");
    end
    endtask

    // ================================================================
    // Main test body
    // ================================================================
    initial begin
        rst_n = 1'b0;
        arm_prog_en = 1'b1; arm_prog_we = 1'b0; arm_dmem_sel = 1'b0;
        gpu_prog_en = 1'b1; gpu_prog_we = 1'b0; gpu_dmem_sel = 1'b0;
        host_byte_addr = 32'h0; host_wdata = 72'h0; host_we = 1'b0;

        $display("\n================================================================");
        $display("  Dual-BRAM Architecture Test: Features + Weights");
        $display("================================================================\n");
        
        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5)  @(posedge clk); #1;

        // ── Program GPU IMEM ──────────────────────────────────────
        // NOTE: Make sure to update instruction 2 and 3 to use your new LDW64 opcode!
        // Using placeholder 0x0E for LDW64.
        $display("[1] Program GPU IMEM (BF_MAC kernel)");
        gpu_imem_write(0, 32'h04400000); // READ_TID R1
        gpu_imem_write(1, 32'h0C800000); // LD64   R2, R0, 0  (Load Feature from 0x000)
        gpu_imem_write(2, 32'h14C00000); // LDW64  R3, R0, 0  (Load Weight from 0x400)
        gpu_imem_write(3, 32'h15000008); // LDW64  R4, R0, 8  (Load Weight from 0x408)
        gpu_imem_write(4, 32'h8548D000); // BF_MAC R5, R2, R3, R4
        gpu_imem_write(5, 32'h11440000); // ST64   R5, R1, 0  (Store Result to 0x000)

        // ── Load Dual BRAM ────────────────────────────────────────
        $display("\n[2] Load Features and Weights via Address Decoder");
        
        // Base 0x000 -> Feature BRAM [cite: 143, 144]
        bram_write_host(32'h000, 64'h4000400040004000); // VecA (Features = 2.0)
        
        // Base 0x400 -> Weight BRAM [cite: 143, 144]
        bram_write_host(32'h400, 64'h3FC03FC03FC03FC0); // VecB (Weights = 1.5)
        bram_write_host(32'h404, 64'h3F003F003F003F00); // VecC (Weights = 0.5)

        // Verify Isolation
        bram_read_host(32'h000, bram_val);
        check64("Feature BRAM[0] verify", bram_val[63:0], 64'h4000400040004000);
        bram_read_host(32'h400, bram_val);
        check64("Weight BRAM[0] verify",  bram_val[63:0], 64'h3FC03FC03FC03FC0);

        // ── Run GPU ───────────────────────────────────────────────
        $display("\n[3] Start GPU execution");
        gpu_prog_en = 1'b0; 
        @(posedge clk); #1; @(posedge clk); #1;

        gpu_start_r = 1'b1;
        @(posedge clk); #1;
        gpu_start_r = 1'b0;

        wait_gpu_done;
        repeat(4) @(posedge clk); #1;

        // ── Check Result ──────────────────────────────────────────
        bram_read_host(32'h000, bram_val);
        check64("GPU BF_MAC Result", bram_val[63:0], 64'h4060406040604060);

        $display("\n================================================================");
        if (fail_count == 0) $display("  ALL TESTS PASSED\n");
        $finish;
    end
endmodule
