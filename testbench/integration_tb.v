`timescale 1ns/1ps
// ============================================================
// integration_tb.v — ModelSim Testbench
//
// Architecture: Separated 3-Module Architecture
// Address Decoder (Bits 11:10):
//   0x000 (2'b00)  →  convertible_fifo (Inputs & Results)
//   0x400 (2'b01)  →  weight_bram      (Weights)
//   0xC00 (2'b11)  →  bias_rom         (Biases)
// ============================================================

module integration_tb;

    // Clock & Reset
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk; // 100MHz clock

    reg rst_n;

    // ARM CPU Wrapper Signals
    reg         arm_prog_we;
    reg         arm_dmem_sel;
    reg  [31:0] arm_prog_addr;
    reg  [31:0] arm_prog_wdata;
    wire [31:0] arm_prog_rdata;
    wire        arm_cpu_done;
    wire        arm_cpu_running;

    // Host registers for simulation
    reg  [31:0] sw_proc_addr;
    reg  [31:0] sw_reg_wdata_hi;
    reg  [31:0] sw_reg_wdata_lo;
    reg         sw_reg_prog_we;

    // Host Read Multiplexer
    wire [71:0] feat_rdata_b;
    wire [71:0] weight_rdata_b;
    wire [71:0] bias_rdata_b;
    reg  [71:0] host_mux_dout;

    // 3-Zone Address Decoder using Bits 11 and 10
    always @(*) begin
        case (sw_proc_addr[11:10])
            2'b00:   host_mux_dout = feat_rdata_b;   // Zone 0: 0x000
            2'b01:   host_mux_dout = weight_rdata_b; // Zone 1: 0x400
            2'b11:   host_mux_dout = bias_rdata_b;   // Zone 3: 0xC00
            default: host_mux_dout = 72'h0;
        endcase
    end

    wire [31:0] sw_reg_rdata_hi = host_mux_dout[63:32];
    wire [31:0] sw_reg_rdata_lo = host_mux_dout[31:0];

    // ARM CPU wrapper signals
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


    // GPU Tensor Core Signals
    reg         gpu_start_r;
    reg         gpu_prog_en;
    reg         gpu_prog_we;
    reg         gpu_dmem_sel;
    reg  [31:0] gpu_prog_addr;
    reg  [31:0] gpu_prog_wdata;
    wire        gpu_done;
    wire        gpu_running;

    wire [7:0]  gpu_feat_addr;
    wire [71:0] gpu_feat_rdata;
    wire [7:0]  gpu_weight_addr;
    wire [71:0] gpu_weight_rdata;
    wire [7:0]  gpu_bias_addr;
    wire [71:0] gpu_bias_rdata;

    wire        gpu_we;
    wire [7:0]  gpu_waddr;
    wire [63:0] gpu_wdata;
    
    assign gpu_waddr = gpu_feat_addr; // Store and Load share the same address bus 

    // Memory Instantiations

    // Convertible FIFO (Wraps feature_bram inside it)
    wire        fifo_we    = gpu_we ? 1'b1 : (sw_reg_prog_we && (sw_proc_addr[11:10] == 2'b00)); 
    wire [7:0]  fifo_addr  = gpu_running ? (gpu_we ? gpu_waddr : gpu_feat_addr) : sw_proc_addr[9:2]; 
    wire [71:0] fifo_wdata = gpu_we ? {8'h0, gpu_wdata} : {8'h0, sw_reg_wdata_hi, sw_reg_wdata_lo};

    convertible_fifo fifo_mem (
        .clk         (clk),
        .reset       (~rst_n),
        .net_in_data (64'h0),
        .net_in_ctrl (8'h0),
        .net_in_wr   (1'b0),
        .net_in_rdy  (),
        .net_out_data(),
        .net_out_ctrl(),
        .net_out_wr  (),
        .net_out_rdy (1'b1),
        .proc_addr   (fifo_addr),
        .proc_wdata  (fifo_wdata),
        .proc_rdata  (feat_rdata_b),
        .proc_we     (fifo_we),
        .mode        (2'b01), // MODE_PROC
        .cmd_send    (1'b0),
        .cmd_reset   (1'b0),
        .pkt_ready   (),
        .pkt_len     (),
        .head_addr   (),
        .tail_addr   ()
    );

    assign gpu_feat_rdata = feat_rdata_b;

    // Weight BRAM
    weight_bram weight_mem (
        .clk      (clk),
        .addr_a   (gpu_weight_addr),
        .rdata_a  (gpu_weight_rdata), 
        .we_b     (sw_reg_prog_we && (sw_proc_addr[11:10] == 2'b01)),
        .addr_b   (sw_proc_addr[9:2]),
        .wdata_b  ({8'h0, sw_reg_wdata_hi, sw_reg_wdata_lo}),
        .rdata_b  (weight_rdata_b)
    );

    // Bias ROM
    bias_rom layer_biases (
        .clk      (clk),
        .addr_a   (gpu_bias_addr),
        .dout_a   (gpu_bias_rdata), 
        .addr_b   (sw_proc_addr[9:2]), 
        .dout_b   (bias_rdata_b)
    );

    // GPU Network Wrapper
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
        .bram_addr      (gpu_feat_addr),
        .bram_wdata     (gpu_wdata),
        .bram_rdata     (gpu_feat_rdata),
        .bram_we        (gpu_we),
        .weight_addr    (gpu_weight_addr),
        .weight_rdata   (gpu_weight_rdata),
        .bias_addr      (gpu_bias_addr),
        .bias_rdata     (gpu_bias_rdata)
    );

    // Simulation & Pre-Flight Test Sequence
    reg [71:0] bram_val;

/*
    initial begin
        // System Initialization
        rst_n = 0;
        sw_proc_addr = 0;
        sw_reg_prog_we = 0;
        
        gpu_start_r = 0;
        gpu_prog_en = 1; // Put GPU in programming mode
        gpu_prog_we = 0;
        gpu_dmem_sel = 0;
        
        #100 rst_n = 1;
        #20;

        $display("\n================================================");
        $display("   PRE-FLIGHT CHECK: PROGRAMMING GPU AND TESTING");
        $display("================================================");

        // [TEST 1] Program the GPU Instruction Memory (IMEM)
        $display("\n[1] Programming GPU Instructions...");
        $readmemh("C:/USC CE/EE533/Final_Project/ANN_NetFPGA-Earthquakes_Detection/sw/gpu_imem.mem", gpu_dut.gpu_core.imem_inst.mem);
        
        // gpu_imem_write(0, 32'h04400000); // READ_TID R1
        // gpu_imem_write(1, 32'h0C800000); // LD64   R2, R0, 0 
        // gpu_imem_write(2, 32'h14C00000); // LDW64  R3, R0, 0 
        // gpu_imem_write(3, 32'h19000000); // LDB64  R4, R0, 0 
        // gpu_imem_write(4, 32'h8548D000); // BF_MAC R5, R2, R3, R4 
        // gpu_imem_write(5, 32'h11440000); // ST64   R5, R1, 0  
        // gpu_imem_write(6, 32'h00000000); // NOP / End Program
        


        // [TEST 2] Verify Bias ROM reads correctly (Address 0xC00)
        $display("\n[2] Verifying Bias ROM at Zone 3 (0xC00)...");
        bram_read_host(32'hC00, bram_val);
        $display("    -> Bias[0] Value: %h", bram_val[63:0]);

        // [TEST 3] Load 1.0 into Feature (0x000) and Weight (0x400)
        $display("\n[3] Loading 1.0 into Feature and Weight BRAMs...");
        bram_write_host(32'h000, 64'h3F803F803F803F80); // Feature = 1.0
        bram_write_host(32'h400, 64'h3F803F803F803F80); // Weight  = 1.0

        // [TEST 4] Trigger GPU
        $display("\n[4] Triggering GPU Execution...");
        gpu_prog_en = 0; // Take GPU out of programming mode!
        @(posedge clk); #1; @(posedge clk); #1;

        // Pulse Start
        gpu_start_r = 1'b1;
        @(posedge clk); #1;
        gpu_start_r = 1'b0;
        
        // Wait for Done
        wait(gpu_done == 1'b1);
        #50; 

        // [TEST 5] Read Result from Convertible FIFO (Address 0x000)
        $display("\n[5] Reading back Result from FIFO BRAM (0x000)...");
        // Because ST64 was to Offset 0 in the instructions, we read from 0x000
        bram_read_host(32'h000, bram_val); 
        
        $display("    -> RESULT  = %h", bram_val[63:0]);
        $display("    -> Does (1.0 * 1.0) + Bias equal the Result? Verify manually.");
        
        $display("\n================================================");
        $display("   END PRE-FLIGHT. READY FOR CSV DATA.");
        $display("================================================");

        #100 $stop;
    end
*/

    initial begin
        // 1. System Initialization 
        rst_n = 0;
        sw_proc_addr = 0;
        sw_reg_prog_we = 0;
        gpu_start_r = 0;
        gpu_prog_en = 1; 
        gpu_prog_we = 0;
        gpu_dmem_sel = 0;
        
        #100 rst_n = 1;
        #20;

        $display("\n================================================");
        $display("   LOADING REAL EARTHQUAKE DATASET INTO GPU");
        $display("================================================");

        // [STEP 1] Load GPU Software (Machine Code)
        // This replaces your manual gpu_imem_write tasks
        $display("\n[1] Loading Compiled ARM CPU and GPU Instructions...");
        $readmemh("../sw/gpu_imem.mem", gpu_dut.gpu_core.imem_inst.mem);
        $readmemh("../sw/imem.mem", arm_dut.arm_inst.imem_inst.mem);

        // [STEP 2] Load Seismic Features (X) into FIFO BRAM
        // This targets the internal RAM inside your convertible_fifo
        $display("[2] Loading Seismic Features from Excel...");
        $readmemh("../dataset/feature_data.mem", fifo_mem.shared_mem.ram);

        // [STEP 3] Load All Weight Layers (W1, W2, W3) into Weight BRAM
        $display("[3] Loading All Weight Layers into Memory...");
        
        // Layer 1: Starts at index 0 (112 lines)
        $readmemh("../dataset/weight_L1.mem", weight_mem.ram, 0);

        // Layer 2: Starts at index 112
        $readmemh("../dataset/weight_L2.mem", weight_mem.ram, 112);

        // Layer 3: Starts after Layer 2
        $readmemh("../dataset/weight_L3.mem", weight_mem.ram, 120);

        // [STEP 4] Trigger GPU Execution
        $display("\n[4] Triggering GPU for Real-Time Inference...");
        gpu_prog_en = 0; // Take GPU out of programming mode
        @(posedge clk); #1; @(posedge clk); #1;

        // Pulse Start
        gpu_start_r = 1'b1;
        @(posedge clk); #1;
        gpu_start_r = 1'b0;
        
        // Monitoring
        // We wait for the hardware to finish the calculation
        wait(gpu_done == 1'b1);
        $display("    -> GPU Inference Complete!");
        #50; 

        // [STEP 5] Read Final Classification Result
        $display("\n[5] Reading Result from Convertible FIFO (0x000)...");
        bram_read_host(32'h000, bram_val); 
        
        $display("    -> RAW HEX RESULT: %h", bram_val[63:0]);
        $display("    -> Verify this against your Python Ground Truth.");

        $display("\n================================================");
        $display("   SIMULATION FINISHED");
        $display("================================================");

        #500 $stop;
    end

    // Host Read/Write Tasks 
    task bram_write_host(input [31:0] addr, input [63:0] data);
    begin
        sw_proc_addr = addr;
        sw_reg_wdata_hi = data[63:32];
        sw_reg_wdata_lo = data[31:0];
        sw_reg_prog_we = 1;
        @ (posedge clk);
        sw_reg_prog_we = 0;
        @ (posedge clk);
    end
    endtask

    task bram_read_host(input [31:0] addr, output [71:0] data);
    begin
        sw_proc_addr = addr;
        @ (posedge clk); 
        @ (posedge clk); // 1 extra cycle for BRAM latency
        data = {8'h0, sw_reg_rdata_hi, sw_reg_rdata_lo};
    end
    endtask

    // GPU IMEM Write Task
    task gpu_imem_write(input [31:0] addr, input [31:0] data);
    begin
        gpu_prog_addr  = addr << 2; // Shift by 2 to align byte addressing for IMEM
        gpu_prog_wdata = data;
        gpu_prog_we    = 1'b1;
        gpu_dmem_sel   = 1'b0;      // 0 routes to IMEM, 1 routes to DMEM
        @ (posedge clk);
        gpu_prog_we    = 1'b0;
        @ (posedge clk);
    end
    endtask

endmodule
