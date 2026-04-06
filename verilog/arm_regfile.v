`timescale 1ns/1ps
// arm_regfile.v — Dual mirrored BRAM, WRITE_FIRST mode
//
// Pre-fetch approach: arm_pipeline presents read addresses during IF stage
// (using if_thread), so data is ready during ID stage.
//
// WB→pre-fetch same-cycle hazard: when WB writes a register at the same
// cycle as the pre-fetch reads it, WRITE_FIRST ensures we get the new value.
// This handles LDR→use patterns across 4-thread interleaving.

module arm_regfile (
    input  wire        clk,
    input  wire        rst,
    input  wire [1:0]  thread_id,   // if_thread (pre-fetch)
    input  wire [1:0]  wb_thread,
    input  wire [3:0]  r0addr,
    output reg  [31:0] r0data,
    input  wire [3:0]  r1addr,
    output reg  [31:0] r1data,
    input  wire        we,
    input  wire [3:0]  waddr,
    input  wire [31:0] wdata,
    input  wire        we2,
    input  wire [3:0]  waddr2,
    input  wire [31:0] wdata2
);
    (* RAM_STYLE = "BLOCK" *) reg [31:0] regs_a [0:63];
    (* RAM_STYLE = "BLOCK" *) reg [31:0] regs_b [0:63];

    integer k;
    initial begin
        for (k = 0; k < 64; k = k + 1) begin
            regs_a[k] = 32'h0;
            regs_b[k] = 32'h0;
        end
        r0data = 32'h0;
        r1data = 32'h0;
    end
    
    wire [5:0] ra0 = {thread_id, r0addr};
    wire [5:0] ra1 = {thread_id, r1addr};
    wire [5:0] wa  = we2 ? {wb_thread, waddr2} : {wb_thread, waddr};
    wire [31:0] wd = we2 ? wdata2 : wdata;
    wire        wen = we | we2;

    // regs_a: WRITE_FIRST — if write and read hit same address same cycle,
    // output the new value (handles WB→pre-fetch same-cycle forwarding)
    always @(posedge clk) begin
        if (wen) begin
            regs_a[wa] <= wd;
            r0data <= (wa == ra0) ? wd : regs_a[ra0];
        end else begin
            r0data <= regs_a[ra0];
        end
    end

    // regs_b: same WRITE_FIRST pattern for r1data
    always @(posedge clk) begin
        if (wen) begin
            regs_b[wa] <= wd;
            r1data <= (wa == ra1) ? wd : regs_b[ra1];
        end else begin
            r1data <= regs_b[ra1];
        end
    end

endmodule
