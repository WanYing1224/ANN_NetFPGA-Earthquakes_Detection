`timescale 1ns/1ps
// ============================================================
// dmem_32.v — ARM Data Memory
//
// 256 x 32-bit true dual-port BRAM.
// (* RAM_STYLE = "BLOCK" *) forces BRAM inference in XST.
//
// Port A  (pipeline — MEM stage)
//   clka / wea / addra / dina / douta
//   Connected to arm_pipeline MEM stage (load/store).
//
// Port B  (host programming interface)
//   clkb / web / addrb / dinb / doutb
//   In Lab 8, Port B is driven by fifo_top's programming MUX:
//     - when prog_en=1, dmem_sel=1: host loads initial data values
//     - after ARM runs: host reads results back via doutb
//   This is the same pattern used in Lab 6 (cpureg.pl / load_and_test.sh).
//
//   Port B address is a 32-bit BYTE address from the SW register;
//   the upper logic in arm_cpu_wrapper shifts [9:2] to get the
//   8-bit word address passed here.
// ============================================================
module dmem_32 (
    // Port A: ARM pipeline MEM stage
    input  wire        clka,
    input  wire        wea,
    input  wire [7:0]  addra,
    input  wire [31:0] dina,
    output reg  [31:0] douta,

    // Port B: host programming / readback interface
    input  wire        clkb,
    input  wire        web,
    input  wire [7:0]  addrb,
    input  wire [31:0] dinb,
    output reg  [31:0] doutb
);
    (* RAM_STYLE = "BLOCK" *)
    reg [31:0] mem [0:255];

    always @(posedge clka) begin
        if (wea) begin
            mem[addra] <= dina;
            douta <= dina;        // write-first: LDR after STR same addr gets new value
        end else begin
            douta <= mem[addra];
        end
    end

    always @(posedge clkb) begin
        if (web) mem[addrb] <= dinb;
        doutb <= mem[addrb];
    end

endmodule
