`timescale 1ns/1ps
// ============================================================
// gpu_ldst.v — GPU Load/Store Unit
// Lab 7: LD/ST unit reads/writes 64-bit from Block RAM
// Lab 8: shared convertible_fifo BRAM is the data source
//
// This unit translates GPU byte-addressed LD64/ST64 instructions
// into the convertible_fifo BRAM word-address interface.
//
// Addressing: byte_addr >> 3 = bram_word_addr (64-bit aligned)
//   bram[0] @ byte 0x00  → Vec A / Result
//   bram[1] @ byte 0x08  → Vec B
//   bram[2] @ byte 0x10  → Vec C
//
// bram_rdata[71:0] = {ctrl[7:0], data[63:0]}
// We use data[63:0] only; ctrl ignored for GPU reads
// ============================================================
module gpu_ldst (
    // LD64: issue read
    input  wire [31:0] ld_byte_addr,
    output wire [7:0]  rd_bram_addr,   // → bram Port B addr
    input  wire [71:0] bram_rdata,     // ← from convertible_fifo
    output wire [63:0] ld_data,        // → register file

    // ST64: issue write
    input  wire [31:0] st_byte_addr,
    input  wire [63:0] st_data,
    output wire [7:0]  wr_bram_addr,
    output wire [71:0] wr_bram_wdata,
    output wire        wr_bram_we,
    input  wire        st_en           // from FSM
);
    // 64-bit word address = byte_addr >> 3
    assign rd_bram_addr = {3'b0, ld_byte_addr[7:3]};
    assign ld_data      = bram_rdata[63:0];  // lower 64 bits = data

    assign wr_bram_addr  = {3'b0, st_byte_addr[7:3]};
    assign wr_bram_wdata = {8'h00, st_data}; // ctrl=0 for GPU writes
    assign wr_bram_we    = st_en;
endmodule
