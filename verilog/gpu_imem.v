`timescale 1ns/1ps
// ============================================================
// gpu_imem.v — GPU Instruction Memory
//
// Port A  (GPU core read — combinational address, sync data)
//   clka / addra[4:0] / douta[31:0]
//   Connected to gpu_core_min fetch stage.
//
// Port B  (host programming — write only)
//   clkb / web / addrb[4:0] / dinb[31:0]
//   Driven by fifo_top when gpu_prog_we=1 && !gpu_dmem_sel.
//
// Depth is 32 words (5-bit PC), matching gpu_core_min's pc[4:0].
// XST infers BRAM from (* RAM_STYLE = "BLOCK" *) + sync read.
// ============================================================
module gpu_imem (
    // Port A: GPU core fetch
    input  wire       clka,
    input  wire [4:0] addra,
    output reg [31:0] douta,

    // Port B: host programming interface
    input  wire       clkb,
    input  wire       web,
    input  wire [4:0] addrb,
    input  wire [31:0] dinb
);
    (* RAM_STYLE = "BLOCK" *)
    reg [31:0] mem [0:31];

    // Port A: synchronous read (required for BRAM inference)
    always @(posedge clka)
        douta <= mem[addra];

    // Port B: synchronous write from host
    always @(posedge clkb)
        if (web) mem[addrb] <= dinb;

endmodule
