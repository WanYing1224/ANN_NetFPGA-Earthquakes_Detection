`timescale 1ns/1ps
// ============================================================
// tensor_unit.v — GPU Tensor Unit (Multi-cycle BF16 MAC)
//
// Updated to match bf16_mac design:
// Uses a 3-stage pipeline (3-cycle latency).
//
// A shift register is used to track pipeline progress,
// generating `busy` and `done` signals to interface cleanly
// with the GPU Core FSM.
// ============================================================

module tensor_unit (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,       // Pulse from FSM EXEC state
    input  wire [63:0] rs1,         // Vector A (4 x BF16)
    input  wire [63:0] rs2,         // Vector B (4 x BF16)
    input  wire [63:0] rs3,         // Vector C (4 x BF16)
    output wire        busy,        // Stall FSM while computation is in progress
    output wire        done,        // Indicates result is ready
    output reg  [63:0] result       // Packed result (4 x BF16)
);

    // Instantiate 4 parallel pipelined BF16 MAC units
    wire [15:0] mac0, mac1, mac2, mac3;

    bf16_mac lane0 (.clk(clk), .a(rs1[15:0]),  .b(rs2[15:0]),  .c(rs3[15:0]),  .z(mac0));
    bf16_mac lane1 (.clk(clk), .a(rs1[31:16]), .b(rs2[31:16]), .c(rs3[31:16]), .z(mac1));
    bf16_mac lane2 (.clk(clk), .a(rs1[47:32]), .b(rs2[47:32]), .c(rs3[47:32]), .z(mac2));
    bf16_mac lane3 (.clk(clk), .a(rs1[63:48]), .b(rs2[63:48]), .c(rs3[63:48]), .z(mac3));

    // ── 3-stage Delay Line (Shift Register) ──
    // Tracks pipeline progress of the MAC operation
    reg [2:0] shift_done;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            shift_done <= 3'b000;
            result     <= 64'd0;
        end else begin
            // Shift the start pulse through the pipeline stages
            shift_done <= {shift_done[1:0], start};
            
            // When the operation reaches stage 2,
            // the MAC outputs will be valid in the next cycle.
            // This register acts as the final pipeline stage (Stage 3 output register).
            if (shift_done[1]) begin
                result <= {mac3, mac2, mac1, mac0};
            end
        end
    end

    // busy:
    // Asserted when a new operation starts or while data is still
    // propagating through the first two pipeline stages.
    // This stalls the GPU FSM during computation.
    assign busy = start | shift_done[0] | shift_done[1];
    
    // done:
    // Asserted when the pipeline reaches the final stage,
    // indicating that the result has been written and is ready.
    assign done = shift_done[2]; 

endmodule
