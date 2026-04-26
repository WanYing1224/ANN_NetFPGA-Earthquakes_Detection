`timescale 1ns/1ps
// ============================================================
// gpu_net.v — Adapter: fifo_top ↔ gpu_core_min 
// ============================================================
module gpu_net (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire        start,
    output wire        done,
    output wire        running,

    // Programming interface
    input  wire        gpu_prog_en,
    input  wire        gpu_prog_we,
    input  wire        gpu_dmem_sel,
    input  wire [31:0] gpu_prog_addr,
    input  wire [31:0] gpu_prog_wdata,

    // Shared Feature BRAM interface
    output wire [7:0]  bram_addr,
    output wire [71:0] bram_wdata,
    input  wire [71:0] bram_rdata,
    output wire        bram_we,

    // Shared Weight BRAM interface
    output wire [7:0]  weight_addr,
    input  wire [71:0] weight_rdata,
    
    // Shared Bias BRAM interface
    output wire [7:0]  bias_addr,
    input  wire [71:0] bias_rdata
);
    wire rst = ~rst_n;

    gpu_core_min gpu_core (
        .clk            (clk),
        .rst            (rst),
        .start          (start),
        .done           (done),
        .running        (running),
        
        // Programming interface
        .gpu_prog_en    (gpu_prog_en),
        .gpu_prog_we    (gpu_prog_we),
        .gpu_dmem_sel   (gpu_dmem_sel),
        .gpu_prog_addr  (gpu_prog_addr),
        .gpu_prog_wdata (gpu_prog_wdata),
        
        // Memory Interfaces
        .bram_addr      (bram_addr),
        .bram_wdata     (bram_wdata),
        .bram_rdata     (bram_rdata),
        .bram_we        (bram_we),
        
        .weight_addr    (weight_addr),
        .weight_rdata   (weight_rdata),
        
        .bias_addr      (bias_addr),
        .bias_rdata     (bias_rdata)
    );

endmodule
