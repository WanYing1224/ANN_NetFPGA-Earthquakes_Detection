`timescale 1ns/1ps
///////////////////////////////////////////////////////////////////////////////
// feature_bram.v - True Dual-Port BRAM for Features / Network Packets
///////////////////////////////////////////////////////////////////////////////

module feature_bram (
    input  wire        clk,
    
    // Port A: Network Side (or GPU)
    input  wire        we_a,
    input  wire [7:0]  addr_a,
    input  wire [71:0] din_a,
    output reg  [71:0] dout_a,
    
    // Port B: Processor Side
    input  wire        we_b,
    input  wire [7:0]  addr_b,
    input  wire [71:0] din_b,
    output reg  [71:0] dout_b
);
    (* RAM_STYLE = "BLOCK" *)
    reg [71:0] ram [0:255];

    // Port A: Synchronous Read/Write
    always @(posedge clk) begin
        if (we_a)
            ram[addr_a] <= din_a;
        dout_a <= ram[addr_a];
    end

    // Port B: Synchronous Read/Write
    always @(posedge clk) begin
        if (we_b)
            ram[addr_b] <= din_b;
        dout_b <= ram[addr_b];
    end
endmodule
