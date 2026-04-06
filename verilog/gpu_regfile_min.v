`timescale 1ns/1ps
// ============================================================
// gpu_regfile_min.v — GPU Register File (8 x 64-bit, BRAM)
// Lab 7: 2+ registers, at least 64-bit wide
//
// Read ports:
//   rs1_out, rs2_out, rs3_out — for ALU / Tensor operands
//   rd_src_out                — 4th port, for ST64 source data
//     (ST64: Mem[Rs_base+Imm] <- Rd, so Rd is a read source)
//
// All reads are synchronous (BRAM inference, saves LUTs)
// Write is synchronous, 1 write port
// R0 hardwired to 0
// ============================================================
module gpu_regfile_min (
    input  wire        clk,
    input  wire        we,
    input  wire [2:0]  rd_addr,
    input  wire [63:0] rd_data,
    input  wire [2:0]  rs1_addr,
    input  wire [2:0]  rs2_addr,
    input  wire [2:0]  rs3_addr,
    input  wire [2:0]  rd_src_addr,  // 4th read port for ST64
    output reg  [63:0] rs1_out,
    output reg  [63:0] rs2_out,
    output reg  [63:0] rs3_out,
    output reg  [63:0] rd_src_out    // ST64 source data
);
    (* RAM_STYLE = "BLOCK" *)
    reg [63:0] regs [0:7];

    always @(posedge clk) begin
        rs1_out    <= (rs1_addr    == 0) ? 64'd0 : regs[rs1_addr];
        rs2_out    <= (rs2_addr    == 0) ? 64'd0 : regs[rs2_addr];
        rs3_out    <= (rs3_addr    == 0) ? 64'd0 : regs[rs3_addr];
        rd_src_out <= (rd_src_addr == 0) ? 64'd0 : regs[rd_src_addr];
    end

    always @(posedge clk) begin
        if (we && rd_addr != 0)
            regs[rd_addr] <= rd_data;
    end
endmodule
