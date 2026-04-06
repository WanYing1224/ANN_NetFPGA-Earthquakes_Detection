`timescale 1ns/1ps
// ============================================================
// fifo_top_tb.v — ModelSim Testbench, Lab 8 Integration
//
// Modified: Now uses the actual `convertible_fifo.v` module 
// instead of a locally simulated shared BRAM array.
// ============================================================

module integration_tb;

    // ----------------------------------------------------------------
    // Clock  (100 MHz, 10 ns period)
    // ----------------------------------------------------------------
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ----------------------------------------------------------------
    // Global reset (active-low)
    // ----------------------------------------------------------------
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
    // GPU signals  (start driven by reg so no force/release needed)
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
        .bram_addr      (gpu_bram_addr),
        .bram_wdata     (gpu_bram_wdata),
        .bram_rdata     (gpu_bram_rdata),
        .bram_we        (gpu_bram_we)
    );

    // ----------------------------------------------------------------
    // Shared FIFO BRAM via convertible_fifo module
    // 
    // We use a multiplexer to share the Processor Port of the 
    // convertible_fifo between the Host and the GPU, just like 
    // in fifo_top.v. When gpu_running=1, the GPU owns the port.
    // ----------------------------------------------------------------
    reg  [7:0]  bram_addr_b;
    reg  [71:0] bram_wdata_b;
    reg         bram_we_b;

    wire [7:0]  mux_addr;
    wire [71:0] mux_wdata;
    wire        mux_we;
    wire [71:0] proc_rdata;

    // Multiplexer priority: GPU > Host
    assign mux_addr  = gpu_running ? gpu_bram_addr  : bram_addr_b;
    assign mux_wdata = gpu_running ? gpu_bram_wdata : bram_wdata_b;
    assign mux_we    = gpu_running ? gpu_bram_we    : bram_we_b;

    // Route read data back to both Host and GPU
    assign gpu_bram_rdata = proc_rdata;
    wire [71:0] bram_rdata_b = proc_rdata;

    convertible_fifo fifo_inst (
        .clk          (clk),
        .reset        (~rst_n),       // convertible_fifo uses active-high reset
        // Network Input Side (tied off for processor tests)
        .net_in_data  (64'b0),
        .net_in_ctrl  (8'b0),
        .net_in_wr    (1'b0),
        .net_in_rdy   (),
        // Network Output Side (tied off)
        .net_out_data (),
        .net_out_ctrl (),
        .net_out_wr   (),
        .net_out_rdy  (1'b1),
        // Processor Side (connected to MUX)
        .proc_addr    (mux_addr),
        .proc_wdata   (mux_wdata),
        .proc_rdata   (proc_rdata),
        .proc_we      (mux_we),
        // Control Interface
        .mode         (2'b01),        // 2'b01 = MODE_PROC
        .cmd_send     (1'b0),
        .cmd_reset    (1'b0),
        .pkt_ready    (),
        .pkt_len      (),
        .head_addr    (),
        .tail_addr    ()
    );

    // ----------------------------------------------------------------
    // Pass / fail counters
    // ----------------------------------------------------------------
    integer pass_count;
    integer fail_count;

    // ----------------------------------------------------------------
    // Loop variable and temp registers
    // ----------------------------------------------------------------
    integer      i;
    reg [31:0]   dmem_val;
    reg [71:0]   bram_val;

    // ================================================================
    // Tasks
    // ================================================================

    // One clock edge + 1 ns setup
    task tick;
    begin
        @(posedge clk); #1;
    end
    endtask

    // ── ARM IMEM write ───────────────────────────────────────────────
    task arm_imem_write;
        input integer word_idx;
        input [31:0]  instr;
    begin
        arm_prog_addr  = word_idx * 4;
        arm_prog_wdata = instr;
        arm_dmem_sel   = 1'b0;        // target IMEM
        @(posedge clk); #1;
        arm_prog_we = 1'b1;
        @(posedge clk); #1;
        arm_prog_we = 1'b0;
    end
    endtask

    // ── ARM DMEM write ───────────────────────────────────────────────
    task arm_dmem_write;
        input integer word_idx;
        input [31:0]  data;
    begin
        arm_prog_addr  = word_idx * 4;
        arm_prog_wdata = data;
        arm_dmem_sel   = 1'b1;        // target DMEM
        @(posedge clk); #1;
        arm_prog_we = 1'b1;
        @(posedge clk); #1;
        arm_prog_we = 1'b0;
    end
    endtask

    // ── ARM DMEM read (two reads for BRAM sync latency, like Lab 6) ──
    task arm_dmem_read;
        input  integer word_idx;
        output [31:0]  rdata;
    begin
        arm_prog_addr = word_idx * 4;
        arm_dmem_sel  = 1'b1;
        @(posedge clk); #1;           // address presented
        @(posedge clk); #1;           // data valid
        rdata = arm_prog_rdata;
    end
    endtask

    // ── GPU IMEM write ───────────────────────────────────────────────
    task gpu_imem_write;
        input integer word_idx;
        input [31:0]  instr;
    begin
        gpu_prog_addr  = word_idx * 4;
        gpu_prog_wdata = instr;
        gpu_dmem_sel   = 1'b0;        // target GPU IMEM
        @(posedge clk); #1;
        gpu_prog_we = 1'b1;
        @(posedge clk); #1;
        gpu_prog_we = 1'b0;
    end
    endtask

    // ── Shared BRAM write (host port B MUX) ──────────────────────────
    task bram_write_host;
        input [7:0]  word_addr;
        input [63:0] data;
    begin
        bram_addr_b  = word_addr;
        bram_wdata_b = {8'h00, data};
        bram_we_b    = 1'b1;
        @(posedge clk); #1;
        bram_we_b    = 1'b0;
    end
    endtask

    // ── Shared BRAM read (host port B MUX, two-cycle latency) ────────
    task bram_read_host;
        input  [7:0]  word_addr;
        output [71:0] rdata;
    begin
        bram_addr_b = word_addr;
        bram_we_b   = 1'b0;
        @(posedge clk); #1;           // address presented
        @(posedge clk); #1;           // data valid
        rdata = bram_rdata_b;
    end
    endtask

    // ── 64-bit check ─────────────────────────────────────────────────
    task check64;
        input [255:0] lbl;
        input [63:0]  got;
        input [63:0]  expected;
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

    // ── Wait for arm_cpu_done (timeout 5000 cycles) ──────────────────
    task wait_arm_done;
        integer n;
    begin
        n = 0;
        while (!arm_cpu_done && n < 5000) begin
            @(posedge clk); #1;
            n = n + 1;
        end
        if (!arm_cpu_done)
            $display("  TIMEOUT: arm_cpu_done never asserted (%0d cycles)", n);
        else
            $display("    arm_cpu_done asserted after %0d cycles", n);
    end
    endtask

    // ── Wait for gpu_done (timeout 10000 cycles) ─────────────────────
    task wait_gpu_done;
        integer n;
    begin
        n = 0;
        while (!gpu_done && n < 10000) begin
            @(posedge clk); #1;
            n = n + 1;
        end
        if (!gpu_done)
            $display("  TIMEOUT: gpu_done never asserted (%0d cycles)", n);
        else
            $display("    gpu_done asserted after %0d cycles", n);
    end
    endtask

    // ================================================================
    // Main test body
    // ================================================================
    initial begin
        // ── Initialise all signals ───────────────────────────────────
        pass_count    = 0;
        fail_count    = 0;
        rst_n         = 1'b0;

        arm_prog_en   = 1'b1;   // ARM held in prog/reset from start
        arm_prog_we   = 1'b0;
        arm_dmem_sel  = 1'b0;
        arm_prog_addr = 32'h0;
        arm_prog_wdata= 32'h0;

        gpu_start_r   = 1'b0;
        gpu_prog_en   = 1'b1;   // GPU held in prog/reset from start
        gpu_prog_we   = 1'b0;
        gpu_dmem_sel  = 1'b0;
        gpu_prog_addr = 32'h0;
        gpu_prog_wdata= 32'h0;

        bram_addr_b   = 8'h0;
        bram_wdata_b  = 72'h0;
        bram_we_b     = 1'b0;

        // ── [1] Reset ────────────────────────────────────────────────
        $display("\n================================================================");
        $display("  Lab 8 Integration TB: ARM CPU + GPU BF16 Tensor Core");
        $display("================================================================\n");
        $display("[1] Reset");
        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5)  @(posedge clk); #1;
        $display("    done\n");

        // ── [2] Program ARM IMEM ─────────────────────────────────────
        $display("[2] Program ARM IMEM");
        arm_imem_write(  0, 32'hE3A00000); // MOV r0, #0
        arm_imem_write(  1, 32'hE3A01038); // MOV r1, #56
        arm_imem_write(  2, 32'hE5903000); // LDR r3, [r0]
        arm_imem_write(  3, 32'hE2833001); // ADD r3, r3, #1
        arm_imem_write(  4, 32'hE5803000); // STR r3, [r0]
        arm_imem_write(  5, 32'hE2800004); // ADD r0, r0, #4
        arm_imem_write(  6, 32'hE1500001); // CMP r0, r1
        arm_imem_write(  7, 32'hBAFFFFF9); // BLT -7 (word 2)
        arm_imem_write(  8, 32'hEAFFFFFE); // B . (halt T0)
        arm_imem_write( 48, 32'hEAFFFFFE); // B . (halt T1)
        arm_imem_write( 96, 32'hEAFFFFFE); // B . (halt T2)
        arm_imem_write(144, 32'hEAFFFFFE); // B . (halt T3)
        $display("    ARM IMEM loaded\n");

        // ── [3] Program ARM DMEM ─────────────────────────────────────
        $display("[3] Program ARM DMEM  (values 0 .. 13)");
        for (i = 0; i < 14; i = i + 1) begin
            arm_dmem_write(i, i[31:0]);
        end
        $display("    ARM DMEM loaded\n");

        // ── [4] Release ARM and run ───────────────────────────────────
        $display("[4] Release ARM and start execution");
        arm_prog_en = 1'b0;    // releases ARM from prog/reset
        arm_prog_we = 1'b0;
        @(posedge clk); #1;

        $display("    waiting for arm_cpu_done...");
        wait_arm_done;

        // ── [4b] Verify ARM DMEM results ─────────────────────────────
        $display("[4b] Verify ARM DMEM results (expect 1 .. 14)");
        arm_prog_en  = 1'b1;   // re-enable prog mode for DMEM readback
        arm_dmem_sel = 1'b1;

        for (i = 0; i < 14; i = i + 1) begin
            arm_dmem_read(i, dmem_val);
            if (dmem_val === (i + 1)) begin
                $display("  PASS  dmem[%0d] == %0d", i, i+1);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL  dmem[%0d]  expected=%0d  got=0x%08h",
                         i, i+1, dmem_val);
                fail_count = fail_count + 1;
            end
        end

        // ── [5] Program GPU IMEM ──────────────────────────────────────
        $display("\n[5] Program GPU IMEM (BF_MAC kernel, 6 instructions)");
        gpu_imem_write(0, 32'h04400000); // READ_TID  R1
        gpu_imem_write(1, 32'h0C800000); // LD64  R2, R0, 0
        gpu_imem_write(2, 32'h0CC00008); // LD64  R3, R0, 8
        gpu_imem_write(3, 32'h0D000010); // LD64  R4, R0, 16
        gpu_imem_write(4, 32'h8548D000); // BF_MAC R5, R2, R3, R4
        gpu_imem_write(5, 32'h11440000); // ST64  R5, R1, 0
        $display("    GPU IMEM loaded\n");

        // ── [6] Load GPU operands into shared BRAM ───────────────────
        $display("[6] Load GPU operands into shared FIFO BRAM");
        $display("    BRAM[0] = VecA: 4 x 2.0  BF16=0x4000 -> 0x4000400040004000");
        $display("    BRAM[1] = VecB: 4 x 1.5  BF16=0x3FC0 -> 0x3FC03FC03FC03FC0");
        $display("    BRAM[2] = VecC: 4 x 0.5  BF16=0x3F00 -> 0x3F003F003F003F00");
        $display("    Expected result: 2.0*1.5+0.5=3.5  BF16=0x4060");
        $display("                     4 lanes: 0x4060406040604060");

        bram_write_host(8'h00, 64'h4000400040004000); // VecA
        bram_write_host(8'h01, 64'h3FC03FC03FC03FC0); // VecB
        bram_write_host(8'h02, 64'h3F003F003F003F00); // VecC

        // Verify BRAM writes before running GPU
        bram_read_host(8'h00, bram_val);
        check64("VecA BRAM[0] write verify", bram_val[63:0], 64'h4000400040004000);

        bram_read_host(8'h01, bram_val);
        check64("VecB BRAM[1] write verify", bram_val[63:0], 64'h3FC03FC03FC03FC0);

        bram_read_host(8'h02, bram_val);
        check64("VecC BRAM[2] write verify", bram_val[63:0], 64'h3F003F003F003F00);

        // ── [7] Release GPU and start ─────────────────────────────────
        $display("\n[7] Release GPU and start execution");
        gpu_prog_en = 1'b0;    // release from prog/reset
        gpu_prog_we = 1'b0;
        @(posedge clk); #1;
        @(posedge clk); #1;    // let reset deassert settle

        // One-cycle start pulse
        gpu_start_r = 1'b1;
        @(posedge clk); #1;
        gpu_start_r = 1'b0;

        $display("    waiting for gpu_done...");
        wait_gpu_done;

        // ── [8] Verify GPU BF16 result ────────────────────────────────
        $display("[8] Verify GPU BF_MAC result at BRAM[0]");
        // Extra cycles for ST64 to complete and BRAM latency to resolve
        repeat(4) @(posedge clk); #1;

        bram_read_host(8'h00, bram_val);
        check64("GPU BF_MAC 2.0*1.5+0.5=3.5", bram_val[63:0], 64'h4060406040604060);

        // ── Summary ───────────────────────────────────────────────────
        $display("\n================================================================");
        $display("  PASS: %0d   FAIL: %0d", pass_count, fail_count);
        $display("================================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED\n");
        else
            $display("  FAILURES detected — inspect waveform for details\n");

        $finish;
    end

    // ----------------------------------------------------------------
    // Watchdog — abort if simulation stalls
    // ----------------------------------------------------------------
    initial begin
        #10_000_000;
        $display("WATCHDOG: simulation exceeded 10 million ns, aborting.");
        $finish;
    end

    // ----------------------------------------------------------------
    // Waveform dump
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("integration_tb_conv.vcd");
        $dumpvars(0, integration_tb);
    end

endmodule
