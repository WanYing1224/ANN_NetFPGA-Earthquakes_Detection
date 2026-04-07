`timescale 1ns/1ps
// ============================================================
// gpu_ldst.v — GPU Load/Store Unit (Dual-BRAM Architecture)
//
// This unit translates GPU byte-addressed LD64/LDW64/ST64 instructions
// into the word-address interface for both the Feature and Weight BRAMs.
// ============================================================
module gpu_ldst (
    // LD64: issue read (Feature BRAM)
    input  wire [31:0] ld_byte_addr,
    output wire [7:0]  rd_bram_addr,   // → Feature BRAM Port B addr
    input  wire [71:0] bram_rdata,     // ← from convertible_fifo
    output wire [63:0] ld_data,        // → register file

    // LDW64: issue read (Weight BRAM - NEW)
    input  wire [31:0] ldw_byte_addr,
    output wire [7:0]  rd_weight_addr, // → Weight BRAM Port A addr
    input  wire [71:0] weight_rdata,   // ← from weight_bram
    output wire [63:0] ldw_data,       // → register file

    // ST64: issue write (Feature BRAM only)
    input  wire [31:0] st_byte_addr,
    input  wire [63:0] st_data,
    output wire [7:0]  wr_bram_addr,
    output wire [71:0] wr_bram_wdata,
    output wire        wr_bram_we,
    input  wire        st_en           // from FSM
);

    // Feature Read (LD64)
    assign rd_bram_addr = {3'b0, ld_byte_addr[7:3]};
    assign ld_data      = bram_rdata[63:0];

    // Weight Read (LDW64)
    assign rd_weight_addr = {3'b0, ldw_byte_addr[7:3]};
    assign ldw_data       = weight_rdata[63:0];

    // Feature Write (ST64)
    assign wr_bram_addr  = {3'b0, st_byte_addr[7:3]};
    assign wr_bram_wdata = {8'h00, st_data}; // ctrl=0 for GPU writes
    assign wr_bram_we    = st_en;

endmodule
