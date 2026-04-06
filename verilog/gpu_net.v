`timescale 1ns/1ps
// ============================================================
// gpu_net.v — Adapter: fifo_top ↔ gpu_core_min 
// ============================================================
module gpu_net (
    input  wire        clk,
    input  wire        rst_n,

    // Control (unchanged)
    input  wire        start,
    output wire        done,
    output wire        running,

    // Programming interface (NEW — from fifo_top SW regs)
    input  wire        gpu_prog_en,
    input  wire        gpu_prog_we,
    input  wire        gpu_dmem_sel,
    input  wire [31:0] gpu_prog_addr,
    input  wire [31:0] gpu_prog_wdata,

    // Shared BRAM interface (unchanged)
    output wire [7:0]  bram_addr,
    output wire [71:0] bram_wdata,
    input  wire [71:0] bram_rdata,
    output wire        bram_we
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
        // Shared BRAM
        .bram_addr      (bram_addr),
        .bram_wdata     (bram_wdata),
        .bram_we        (bram_we),
        .bram_rdata     (bram_rdata)
    );

endmodule
