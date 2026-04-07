// ============================================================
// xilinx_primitives_sim.v
//
// ModelSim simulation stubs for Xilinx Virtex-II Pro primitives
// used in bf16_mac.v (structural synthesis version).
//
// These modules are NOT used for synthesis — ISE resolves them
// from the real unisim library. They are ONLY compiled for
// ModelSim so the structural bf16_mac.v simulates correctly.
//
// Compile order:
//   1. vlog xilinx_primitives_sim.v
//   2. vlog bf16_mac.v tensor_unit.v ... (all other sources)
//   3. vlog fifo_top_tb.v
//   4. vsim fifo_top_tb
//
// Primitives modelled:
//   MULT18X18S — 18×18 signed registered multiplier (DSP18)
//   MUXCY      — carry chain propagate/generate mux
//   XORCY      — carry chain sum bit
//   LUT2       — 2-input look-up table (parameterised by INIT)
//   MUXF5      — dedicated 2:1 F5 mux within slice
// ============================================================
`timescale 1ns/1ps

// ── MULT18X18S ───────────────────────────────────────────────
// 18×18 signed multiplier with registered output.
// C  = clock
// CE = clock enable (active high)
// R  = synchronous reset (active high)
// A  = 18-bit signed input
// B  = 18-bit signed input
// P  = 36-bit signed product (registered)
//
// In bf16_mac.v we zero-pad 8-bit unsigned mantissas to 18 bits
// ({10'b0, mant}) so the signed arithmetic gives unsigned result.
module MULT18X18S (
    input  wire        C,
    input  wire        CE,
    input  wire        R,
    input  wire [17:0] A,
    input  wire [17:0] B,
    output reg  [35:0] P
);
    always @(posedge C) begin
        if (R)
            P <= 36'b0;
        else if (CE)
            P <= $signed(A) * $signed(B);
    end
endmodule

// ── MUXCY ────────────────────────────────────────────────────
// Carry chain propagate/generate mux.
// O  = carry out
// CI = carry in
// DI = data in (generate value when S=0)
// S  = select: 1 = propagate CI, 0 = use DI
//
// Logic: O = S ? CI : DI
module MUXCY (
    output wire O,
    input  wire CI,
    input  wire DI,
    input  wire S
);
    assign O = S ? CI : DI;
endmodule

// ── XORCY ────────────────────────────────────────────────────
// Carry chain sum bit.
// O  = sum output
// CI = carry in
// LI = local input (half-sum = A XOR B before carry)
//
// Logic: O = CI XOR LI
module XORCY (
    output wire O,
    input  wire CI,
    input  wire LI
);
    assign O = CI ^ LI;
endmodule

// ── LUT2 ─────────────────────────────────────────────────────
// 2-input look-up table.
// INIT = 4-bit truth table: INIT[{I1,I0}] = output
//
// Common INIT values used in bf16_mac.v:
//   4'h5  = NOT:  O = ~I0        (0101)
//   4'h6  = XOR:  O = I0 ^ I1   (0110)
//   4'h8  = AND:  O = I0 & I1   (1000)
//   4'hE  = OR:   O = I0 | I1   (1110)
module LUT2 #(
    parameter INIT = 4'h0
) (
    output wire O,
    input  wire I0,
    input  wire I1
);
    assign O = INIT[{I1, I0}];
endmodule

// ── MUXF5 ────────────────────────────────────────────────────
// Dedicated F5 mux within Xilinx slice (0 LUTs).
// O  = output
// I0 = input when S=0
// I1 = input when S=1
// S  = select
//
// Logic: O = S ? I1 : I0
module MUXF5 (
    output wire O,
    input  wire I0,
    input  wire I1,
    input  wire S
);
    assign O = S ? I1 : I0;
endmodule
