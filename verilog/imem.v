`timescale 1ns/1ps
// ============================================================
// imem.v — ARM Instruction Memory 
//
// Port A  (pipeline read, combinational address → sync data)
//   clka  : pipeline clock
//   addra : 9-bit word address from IF stage  (pc[if_thread])
//   douta : 32-bit instruction output (1-cycle BRAM latency)
//   data_addr / data_dout : secondary read port for PC-relative LDR
//   wea / dina : UNUSED on Port A in normal operation
//
// Port B  (programming interface, write-only from host)
//   clkb  : same clock
//   web   : write enable — asserted by fifo_top when prog_we=1 && !dmem_sel
//   addrb : 9-bit word address from prog_addr SW register
//   dinb  : 32-bit instruction word from prog_wdata SW register
//
// XST infers RAMB16 from the (* RAM_STYLE = "BLOCK" *) attribute
// and the synchronous read template.
// ============================================================
module imem (
    // Port A: instruction fetch (pipeline)
    input  wire        clka,
    input  wire [8:0]  addra,
    output reg  [31:0] douta,
    // secondary read port (PC-relative LDR through EX/MEM stage)
    input  wire [8:0]  data_addr,
    output reg  [31:0] data_dout,
    // Port A write (unused in normal operation, kept for interface compat)
    input  wire        wea,
    input  wire [31:0] dina,

    // Port B: host programming interface
    input  wire        clkb,
    input  wire        web,
    input  wire [8:0]  addrb,
    input  wire [31:0] dinb
);

    (* RAM_STYLE = "BLOCK" *)
    reg [31:0] mem [0:511];   // 512 x 32-bit = 2 KB

    // Port A: synchronous read (BRAM inference requires sync read)
    always @(posedge clka) begin
        if (wea)
            mem[addra] <= dina;
        douta     <= mem[addra];
        data_dout <= mem[data_addr];
    end

    // Port B: synchronous write from host programming interface
    always @(posedge clkb) begin
        if (web)
            mem[addrb] <= dinb;
    end

endmodule
