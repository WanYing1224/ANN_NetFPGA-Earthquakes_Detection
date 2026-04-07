// ============================================================
// arm_alu.v  —  32-bit Combinational ALU
//
// Pure combinational — no clock, no registers.
// The pipeline register (EX/MEM) latches the output.
//
// op encoding:
//   000 = ADD   A + B
//   001 = SUB   A - B
//   010 = AND   A & B
//   011 = OR    A | B
//   100 = XOR   A ^ B
//   101 = MOV   B  (pass second operand through)
//   110 = reserved (treated as ADD)
//   111 = reserved (treated as ADD)
//
// Flags (combinational, for CPSR update):
//   N = result[31]
//   Z = (result == 0)
//   C = carry out (ADD) or NOT borrow (SUB)
//   V = signed overflow
//
// Verilog-2001 compatible (ISE 10.1).
// ============================================================
module arm_alu (
    input  wire [31:0] A,      // Rn (first operand)
    input  wire [31:0] B,      // shifted Op2 (second operand)
    input  wire [2:0]  op,

    output reg  [31:0] result,
    output wire        N,
    output wire        Z,
    output reg         C,
    output reg         V
);

    // 33-bit extended result for carry detection
    reg [32:0] ext;

    always @(*) begin
        ext    = 33'b0;
        result = 32'b0;
        C      = 1'b0;
        V      = 1'b0;

        case (op)
            3'b000: begin   // ADD
                ext    = {1'b0, A} + {1'b0, B};
                result = ext[31:0];
                C      = ext[32];
                // Overflow: same-sign inputs, different-sign output
                V = (A[31] == B[31]) && (result[31] != A[31]);
            end
            3'b001: begin   // SUB
                ext    = {1'b0, A} - {1'b0, B};
                result = ext[31:0];
                // ARM: C = NOT borrow (C=1 when A >= B, no borrow)
                C      = (A >= B) ? 1'b1 : 1'b0;
                // Overflow: different-sign inputs, wrong-sign output
                V = (A[31] != B[31]) && (result[31] != A[31]);
            end
            3'b010: result = A & B;         // AND
            3'b011: result = A | B;         // OR
            3'b100: result = A ^ B;         // XOR
            3'b101: result = B;             // MOV (pass B through)
            default: begin                  // default to ADD
                ext    = {1'b0, A} + {1'b0, B};
                result = ext[31:0];
                C      = ext[32];
                V = (A[31] == B[31]) && (result[31] != A[31]);
            end
        endcase
    end

    assign N = result[31];
    assign Z = (result == 32'b0);

endmodule
