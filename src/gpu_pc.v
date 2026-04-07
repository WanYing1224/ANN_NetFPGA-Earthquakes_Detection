`timescale 1ns/1ps
// ============================================================
// gpu_pc.v — GPU Program Counter
// Lab 7: single program counter, supports pc+1 / branch / halt
// ============================================================
module gpu_pc (
    input  wire       clk,
    input  wire       rst,
    input  wire       stall,        // hold PC (e.g. tensor busy)
    input  wire       branch_en,    // take branch
    input  wire [4:0] branch_target,// branch destination
    input  wire       halt,         // stop advancing
    output reg  [4:0] pc
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            pc <= 5'd0;
        else if (halt || stall)
            pc <= pc;               // hold
        else if (branch_en)
            pc <= branch_target;
        else
            pc <= pc + 5'd1;
    end
endmodule
