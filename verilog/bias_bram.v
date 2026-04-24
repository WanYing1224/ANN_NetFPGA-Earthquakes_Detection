`timescale 1ns/1ps
// ===============================================================
// bias_rom.v — Dedicated ROM for Biases (With Host Readback)
// 64-bit data (4 x BFloat16 lanes), padded to 72-bit
// ===============================================================

module bias_bram (
    input  wire        clk,
    
    // Port A: GPU Read Path (Runtime)
    input  wire [7:0]  addr_a,
    output reg  [71:0] dout_a,
    
    // Port B: Host Read Path (Testbench Verification)
    input  wire [7:0]  addr_b,
    output reg  [71:0] dout_b
);

    reg [71:0] rom_array [0:15];

    initial begin
        // --- Layer 1 Biases (B1.csv) ---
        rom_array[8'd0] = 72'h00_3EC8_3F01_3E85_BEDA;
        rom_array[8'd1] = 72'h00_3E95_3E4E_3EE2_3E9E;
        rom_array[8'd2] = 72'h00_3EB0_3EC3_3E60_3EE2;
        rom_array[8'd3] = 72'h00_3EC4_3F16_3F00_BEDD;

        // --- Layer 2 Biases (B2.csv) ---
        rom_array[8'd4] = 72'h00_BD67_3E41_3EAC_BD50;
        rom_array[8'd5] = 72'h00_BD25_3CA9_3EC0_3E79;

        // --- Layer 3 Bias (B3.csv) ---
        rom_array[8'd6] = 72'h00_0000_0000_0000_BD8C;
        
        // Zero out remaining just to be safe
        rom_array[8'd7] = 72'h0;
        rom_array[8'd8] = 72'h0;
    end

    always @(posedge clk) begin
        dout_a <= rom_array[addr_a];
        dout_b <= rom_array[addr_b]; // Allows Testbench to verify!
    end

endmodule
