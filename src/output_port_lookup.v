///////////////////////////////////////////////////////////////////////////////
// output_port_lookup.v — Passthrough replacement (Lab 10)
//
// The original output_port_lookup performs MAC address lookup and writes
// an output port bitmask into each packet header. This costs 3,000-6,000
// LUTs for the hash table and ARP state machine.
//
// For Lab 10 demo purposes, we do not need real MAC/port lookup.
// Packets only need to flow through to fifo_top for ARM+GPU processing.
// This module replaces the full implementation with a wire passthrough:
//
//   Data path:   in_data/ctrl/wr → out_data/ctrl/wr  (direct assign)
//   Backpressure: out_rdy → in_rdy                   (direct assign)
//   Register bus: reg_*_in → reg_*_out               (chain passthrough)
//   clk, reset:  accepted but unused
//
// Port interface is IDENTICAL to the original output_port_lookup —
// user_data_path.v instantiation is unchanged.
//
// LUT usage: 0 (pure wire connections)
// FF usage:  0
///////////////////////////////////////////////////////////////////////////////
`timescale 1ns/1ps

module output_port_lookup
  #(parameter DATA_WIDTH        = 64,
    parameter CTRL_WIDTH        = DATA_WIDTH/8,
    parameter UDP_REG_SRC_WIDTH = 2,
    parameter INPUT_ARBITER_STAGE_NUM = 2,
    parameter STAGE_NUM         = 4,
    parameter NUM_OUTPUT_QUEUES = 8,
    parameter NUM_IQ_BITS       = 3)

   (// --- data path ---
    output wire [DATA_WIDTH-1:0]            out_data,
    output wire [CTRL_WIDTH-1:0]            out_ctrl,
    output wire                             out_wr,
    input  wire                             out_rdy,

    input  wire [DATA_WIDTH-1:0]            in_data,
    input  wire [CTRL_WIDTH-1:0]            in_ctrl,
    input  wire                             in_wr,
    output wire                             in_rdy,

    // --- register bus (chain passthrough) ---
    input  wire                             reg_req_in,
    input  wire                             reg_ack_in,
    input  wire                             reg_rd_wr_L_in,
    input  wire [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_in,
    input  wire [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_in,
    input  wire [UDP_REG_SRC_WIDTH-1:0]     reg_src_in,

    output wire                             reg_req_out,
    output wire                             reg_ack_out,
    output wire                             reg_rd_wr_L_out,
    output wire [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_out,
    output wire [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_out,
    output wire [UDP_REG_SRC_WIDTH-1:0]     reg_src_out,

    input  wire                             clk,
    input  wire                             reset
   );

   // ── Data path: direct passthrough ────────────────────────────
   assign out_data = in_data;
   assign out_ctrl = in_ctrl;
   assign out_wr   = in_wr;
   assign in_rdy   = out_rdy;

   // ── Register bus: chain passthrough ──────────────────────────
   assign reg_req_out      = reg_req_in;
   assign reg_ack_out      = reg_ack_in;
   assign reg_rd_wr_L_out  = reg_rd_wr_L_in;
   assign reg_addr_out     = reg_addr_in;
   assign reg_data_out     = reg_data_in;
   assign reg_src_out      = reg_src_in;

endmodule
