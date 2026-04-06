`timescale 1ns/1ps
// Dedicated 256 x 72-bit BRAM for ANN Weights
module weight_bram (
    input  wire        clk,
    // Port A: GPU Read-Only (Runtime)
    input  wire [7:0]  addr_a,
    output reg  [71:0] rdata_a,
    
    // Port B: Host Read/Write (Programming)
    input  wire        we_b,
    input  wire [7:0]  addr_b,
    input  wire [71:0] wdata_b,
    output reg  [71:0] rdata_b
);
    (* RAM_STYLE = "BLOCK" *)
    reg [71:0] ram [0:255];

    always @(posedge clk) begin
        rdata_a <= ram[addr_a]; // GPU fetches weight
        
        if (we_b) ram[addr_b] <= wdata_b;
        rdata_b <= ram[addr_b]; // Host readback
    end
endmodule
