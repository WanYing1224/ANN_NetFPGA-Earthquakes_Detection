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

module integration2_tb;

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

/*
// Testbench Variables for layer 1 Network Streaming
    integer i, j;
    integer fd; // File Descriptor for exporting results
    reg [63:0] full_dataset [0:11549];
    reg [71:0] bram_val;

    initial begin
        // ... (System Reset & Initial Loading logic stays the same) ...

        // 1. Load the Layer 1 ONLY GPU instructions
        $readmemh("../sw/gpu_imem_L1.mem", gpu_dut.gpu_core.imem_inst.mem);
        
        // 2. Load the features and Layer 1 Weights
        $readmemh("../dataset/weight_L1.mem", weight_mem.ram, 0);
        $readmemh("../dataset/feature_data.mem", full_dataset);

        // Take GPU out of programming mode
        gpu_prog_en = 0; 
        @(posedge clk); #1; @(posedge clk); #1;

        $display("\n[+] Starting Layer 1 Execution...");
        
        // OPEN THE OUTPUT FILE
        fd = $fopen("layer1_result.hex", "w");
        if (fd == 0) begin
            $display("ERROR: Could not open layer1_result.hex for writing!");
            $stop;
        end

        // --- BATCH INFERENCE LOOP ---
        for (i = 0; i < 1650; i = i + 1) begin
            
            // A. STREAM IN: Write 7 lines (28 features) to GPU Address 0x000
            for (j = 0; j < 7; j = j + 1) begin
                // NOTE: Using (j * 4) to correctly align with your address decoder
                bram_write_host((j * 4), full_dataset[(i * 7) + j]); 
            end
            
            // B. COMPUTE: Handshake Trigger
            gpu_start_r = 1'b1;
            
            // Wait for GPU to transition out of S_IDLE
            wait(gpu_dut.gpu_core.running == 1'b1); 
            @(posedge clk); #1;
            gpu_start_r = 1'b0; // Drop the start signal
            
            // Wait for Layer 1 execution to finish
            wait(gpu_done == 1'b1);
            
            // C. STREAM OUT & EXPORT: Read the 4 result lines and write to file
            for (j = 0; j < 4; j = j + 1) begin
                bram_read_host((j * 4), bram_val); 
                // Write exactly 16 hex characters (64 bits) to the file
                $fdisplay(fd, "%016x", bram_val[63:0]);
            end
            
            if (i % 100 == 0) begin
                $display("    -> Processed Layer 1 for Event %0d", i);
            end
            
            #20; // Short gap to allow hardware FSMs to settle
        end

        // CLOSE THE OUTPUT FILE
        $fclose(fd);

        $display("\n================================================");
        $display("   LAYER 1 COMPLETE: Output saved to layer1_result.hex");
        $display("================================================");

        #500 $stop;
    end
*/

/*
// Testbench Variables for Phase 2
    integer i, j;
    integer fd; 
    reg [63:0] l1_dataset [0:6599]; // Holds the 6,600 lines from Phase 1
    reg [71:0] bram_val;

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

        // 2. Load the Layer 2 ONLY GPU instructions
        $readmemh("../sw/gpu_imem_L2.mem", gpu_dut.gpu_core.imem_inst.mem);
        
        // 3. Load Layer 2 Weights and the Phase 1 Results
        // Ensure this index (112) matches the offset you compiled into the assembly!
        $readmemh("../dataset/weight_L2.mem", weight_mem.ram, 112); 
        $readmemh("layer1_result.hex", l1_dataset);

        // Take GPU out of programming mode
        gpu_prog_en = 0; 
        @(posedge clk); #1; @(posedge clk); #1;

        $display("\n[+] Starting Layer 2 Execution...");
        
        // OPEN THE OUTPUT FILE
        fd = $fopen("layer2_result.hex", "w");
        if (fd == 0) begin
            $display("ERROR: Could not open layer2_result.hex for writing!");
            $stop;
        end

        // --- BATCH INFERENCE LOOP ---
        for (i = 0; i < 1650; i = i + 1) begin
            
            // A. STREAM IN: Write 4 lines (16 features) to GPU Address 0x000
            for (j = 0; j < 4; j = j + 1) begin
                bram_write_host((j * 4), l1_dataset[(i * 4) + j]); 
            end
            
            // B. COMPUTE: Handshake Trigger
            gpu_start_r = 1'b1;
            
            wait(gpu_dut.gpu_core.running == 1'b1); 
            @(posedge clk); #1;
            gpu_start_r = 1'b0; 
            
            wait(gpu_done == 1'b1);
            
            // C. STREAM OUT & EXPORT: Read the 1 result line and write to file
            bram_read_host(0, bram_val); 
            $fdisplay(fd, "%016x", bram_val[63:0]);
            
            if (i % 100 == 0) begin
                $display("    -> Processed Layer 2 for Event %0d", i);
            end
            
            #20; 
        end

        // CLOSE THE OUTPUT FILE
        $fclose(fd);

        $display("\n================================================");
        $display("   LAYER 2 COMPLETE: Output saved to layer2_result.hex");
        $display("================================================");

        #500 $stop;
    end
*/

// Master Testbench Variables
    integer i, j;
    integer fd; 
    reg [71:0] bram_val;
    
    // Arrays strictly dimensioned for the exact network shapes
    reg [63:0] raw_dataset [0:11549]; // Phase 0: 28 features (7 lines)
    reg [63:0] l1_dataset  [0:6599];  // Phase 1: 16 features (4 lines)
    reg [63:0] l2_dataset  [0:3299];  // Phase 2: 8 features (2 lines)

    initial begin
        rst_n = 0;
        sw_proc_addr = 0;
        sw_reg_prog_we = 0;
        gpu_start_r = 0;
        gpu_prog_en = 1; 
        gpu_prog_we = 0;
        gpu_dmem_sel = 0;
        #100 rst_n = 1; #20;

        $display("\n================================================");
        $display("   STARTING FILE-BASED HOST/FPGA SIMULATION");
        $display("================================================\n");

        // ==========================================
        // PHASE 1 (1650 Events -> 16 Features)
        // ==========================================
        $display("[+] Phase 1: Exporting to layer1_result.hex...");
        $readmemh("../sw/gpu_imem_L1.mem", gpu_dut.gpu_core.imem_inst.mem); 
        $readmemh("../dataset/weight_L1.mem", weight_mem.ram, 0);
        $readmemh("../dataset/feature_data.mem", raw_dataset);

        gpu_prog_en = 0; 
        @(posedge clk); #1; @(posedge clk); #1;
        
        fd = $fopen("layer1_result.hex", "w");
        for (i = 0; i < 1650; i = i + 1) begin
            for (j = 0; j < 7; j = j + 1) bram_write_host((j * 4), raw_dataset[(i * 7) + j]); 
            gpu_start_r = 1'b1; wait(gpu_dut.gpu_core.running == 1'b1); 
            @(posedge clk); #1; gpu_start_r = 1'b0; wait(gpu_done == 1'b1);
            for (j = 0; j < 4; j = j + 1) begin
                bram_read_host((j * 4), bram_val); 
                $fdisplay(fd, "%016x", bram_val[63:0]); 
            end
        end
        $fclose(fd); 

        // ==========================================
        // PHASE 2 (1650 Events -> 8 Features)
        // ==========================================
        $display("[+] Phase 2: Exporting to layer2_result.hex...");
        gpu_prog_en = 1; @(posedge clk); #1;
        
        $readmemh("layer1_result.hex", l1_dataset);
        $readmemh("../sw/gpu_imem_L2.mem", gpu_dut.gpu_core.imem_inst.mem); 
        $readmemh("../dataset/weight_L2.mem", weight_mem.ram, 112);
        gpu_prog_en = 0; @(posedge clk); #1;
        
        fd = $fopen("layer2_result.hex", "w");
        for (i = 0; i < 1650; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) bram_write_host((j * 4), l1_dataset[(i * 4) + j]); 
            gpu_start_r = 1'b1; wait(gpu_dut.gpu_core.running == 1'b1); 
            @(posedge clk); #1; gpu_start_r = 1'b0; wait(gpu_done == 1'b1);
            
            // FIXED: Reading 2 lines (8 features) for Layer 2!
            for (j = 0; j < 2; j = j + 1) begin
                bram_read_host((j * 4), bram_val); 
                $fdisplay(fd, "%016x", bram_val[63:0]); 
            end
        end
        $fclose(fd);

        // ==========================================
        // PHASE 3 (Final Classification)
        // ==========================================
        $display("[+] Phase 3: Exporting to layer3_result.hex...");
        gpu_prog_en = 1; @(posedge clk); #1;
        
        $readmemh("layer2_result.hex", l2_dataset);
        $readmemh("../sw/gpu_imem_L3.mem", gpu_dut.gpu_core.imem_inst.mem); 
        $readmemh("../dataset/weight_L3.mem", weight_mem.ram, 120);
        gpu_prog_en = 0; @(posedge clk); #1;
        
        fd = $fopen("layer3_result.hex", "w");
        for (i = 0; i < 1650; i = i + 1) begin
            
            // FIXED: Streaming in 2 lines (8 features) for Layer 3!
            for (j = 0; j < 2; j = j + 1) begin
                bram_write_host((j * 4), l2_dataset[(i * 2) + j]); 
            end
            
            gpu_start_r = 1'b1; wait(gpu_dut.gpu_core.running == 1'b1); 
            @(posedge clk); #1; gpu_start_r = 1'b0; wait(gpu_done == 1'b1);
            
            bram_read_host(0, bram_val); 
            $fdisplay(fd, "%016x", bram_val[63:0]); 
            
            if (i % 100 == 0) $display("    -> Event %0d Final Result: %h", i, bram_val[63:0]);
        end
        $fclose(fd);

        $display("\n================================================");
        $display("   ALL PHASES COMPLETE. CHECK .HEX FILES.");
        $display("================================================\n");

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
