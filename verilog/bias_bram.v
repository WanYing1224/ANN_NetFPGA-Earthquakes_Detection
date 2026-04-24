`timescale 1ns/1ps
// ===============================================================
// bias_bram.v — Dedicated Bias ROM for GPU LDB64
//
// Board-safe version:
//   - Does not rely on an initial block / inferred memory init.
//   - Uses explicit constant case statements so XST synthesizes the
//     bias values directly into logic/ROM.
//   - Port A: GPU runtime read.
//   - Port B: host readback through fifo_top.
// ===============================================================

module bias_bram (
    input  wire        clk,

    // Port A: GPU Read Path (Runtime)
    input  wire [7:0]  addr_a,
    output reg  [71:0] dout_a,

    // Port B: Host Read Path (Verification)
    input  wire [7:0]  addr_b,
    output reg  [71:0] dout_b
);

    function [71:0] bias_word;
        input [7:0] addr;
        begin
            case (addr)
                // Layer 1 Biases
                8'd0: bias_word = 72'h00_3EC8_3F01_3E85_BEDA;
                8'd1: bias_word = 72'h00_3E95_3E4E_3EE2_3E9E;
                8'd2: bias_word = 72'h00_3EB0_3EC3_3E60_3EE2;
                8'd3: bias_word = 72'h00_3EC4_3F16_3F00_BEDD;

                // Layer 2 Biases
                8'd4: bias_word = 72'h00_BD67_3E41_3EAC_BD50;
                8'd5: bias_word = 72'h00_BD25_3CA9_3EC0_3E79;

                // Layer 3 Bias
                8'd6: bias_word = 72'h00_0000_0000_0000_BD8C;

                default: bias_word = 72'h0;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        dout_a <= bias_word(addr_a);
        dout_b <= bias_word(addr_b);
    end

endmodule
