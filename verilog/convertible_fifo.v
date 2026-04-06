`timescale 1ns/1ps
///////////////////////////////////////////////////////////////////////////////
// convertible_fifo.v - Convertible FIFO (Refactored)
//
// Separated the memory array into 'feature_bram'. This module now only
// handles the Network packet capture/replay logic and mode switching.
///////////////////////////////////////////////////////////////////////////////

module convertible_fifo (
    input  wire        clk,
    input  wire        reset,

    // === Network Input Side ===
    input  wire [63:0] net_in_data,
    input  wire [7:0]  net_in_ctrl,
    input  wire        net_in_wr,
    output wire        net_in_rdy,

    // === Network Output Side ===
    output reg  [63:0] net_out_data,
    output reg  [7:0]  net_out_ctrl,
    output reg         net_out_wr,
    input  wire        net_out_rdy,

    // === Processor Side ===
    input  wire [7:0]  proc_addr,
    input  wire [71:0] proc_wdata,
    output wire [71:0] proc_rdata,    // 1-cycle latency (synchronous)
    input  wire        proc_we,

    // === Control Interface ===
    input  wire [1:0]  mode,
    input  wire        cmd_send,
    input  wire        cmd_reset,
    output wire        pkt_ready,
    output wire [7:0]  pkt_len,
    output wire [7:0]  head_addr,
    output wire [7:0]  tail_addr
);

    // ----- Mode definitions -----
    localparam MODE_IDLE     = 2'b00;
    localparam MODE_PROC     = 2'b01;
    localparam MODE_FIFO_OUT = 2'b10;
    localparam MODE_FIFO_IN  = 2'b11;

    // =========================================================
    // External BRAM Instantiation (Replaced internal reg array)
    // =========================================================
    reg [7:0]  porta_addr;
    wire [71:0] porta_rdata; // Changed from reg to wire to receive from module
    reg        porta_we;
    reg [71:0] porta_wdata;
    
    // Gate the processor write-enable by checking the mode
    wire proc_we_gated = proc_we && (mode == MODE_PROC);

    feature_bram shared_mem (
        .clk    (clk),
        
        // Port A (Network/FIFO control)
        .we_a   (porta_we),
        .addr_a (porta_addr),
        .din_a  (porta_wdata),
        .dout_a (porta_rdata),
        
        // Port B (Processor Control)
        .we_b   (proc_we_gated),
        .addr_b (proc_addr),
        .din_b  (proc_wdata),
        .dout_b (proc_rdata)
    );

    // ----- FIFO Pointers -----
    reg [7:0] head;
    reg [7:0] tail;
    reg [7:0] word_count;
    reg       pkt_ready_r;

    assign head_addr = head;
    assign tail_addr = tail;
    assign pkt_len   = word_count;
    assign pkt_ready = pkt_ready_r;

    // ----- Packet capture state machine -----
    localparam CAP_IDLE = 2'b00;
    localparam CAP_RECV = 2'b01;
    localparam CAP_DONE = 2'b10;

    reg [1:0] cap_state;

    assign net_in_rdy = (mode == MODE_FIFO_IN) && (cap_state != CAP_DONE)
                        && (tail < 8'd250);

    // ----- FIFO Output state machine -----
    localparam OUT_IDLE    = 2'b00;
    localparam OUT_READING = 2'b01;  
    localparam OUT_SENDING = 2'b10;
    localparam OUT_DONE    = 2'b11;

    reg [1:0] out_state;
    reg [7:0] out_rd_addr;   

    // ----- Main Control Logic -----
    always @(posedge clk) begin
        if (reset || cmd_reset) begin
            head         <= 8'd0;
            tail         <= 8'd0;
            word_count   <= 8'd0;
            pkt_ready_r  <= 1'b0;
            cap_state    <= CAP_IDLE;
            net_out_data <= 64'b0;
            net_out_ctrl <= 8'b0;
            net_out_wr   <= 1'b0;
            out_state    <= OUT_IDLE;
            out_rd_addr  <= 8'd0;
            porta_we     <= 1'b0;
            porta_addr   <= 8'd0;
            porta_wdata  <= 72'b0;
        end else begin

            // Default: no write on port A
            porta_we <= 1'b0;

            // =============================================
            // FIFO Input: Capture packet word by word
            // =============================================
            if (mode == MODE_FIFO_IN && net_in_wr && cap_state != CAP_DONE) begin
                porta_addr  <= tail;
                porta_wdata <= {net_in_ctrl, net_in_data};
                porta_we    <= 1'b1;
                tail        <= tail + 1;
                word_count  <= word_count + 1;

                case (cap_state)
                    CAP_IDLE: begin
                        if (net_in_ctrl != 8'b0) begin
                            word_count <= 8'd1;
                            cap_state  <= CAP_RECV;
                        end
                    end
                    CAP_RECV: begin
                        if (net_in_ctrl != 8'b0) begin
                            cap_state   <= CAP_DONE;
                            pkt_ready_r <= 1'b1;
                        end
                    end
                    default: ;
                endcase
            end

            // =============================================
            // FIFO Output: Replay packet to network
            // =============================================
            if (mode == MODE_FIFO_OUT) begin
                case (out_state)
                    OUT_IDLE: begin
                        net_out_wr <= 1'b0;
                        if (word_count > 0) begin
                            porta_addr  <= head;
                            out_rd_addr <= head;
                            out_state   <= OUT_READING;
                        end
                    end

                    OUT_READING: begin
                        net_out_wr <= 1'b0;
                        out_state  <= OUT_SENDING;
                    end

                    OUT_SENDING: begin
                        if (net_out_rdy) begin
                            net_out_data <= porta_rdata[63:0];
                            net_out_ctrl <= porta_rdata[71:64];
                            net_out_wr   <= 1'b1;
                            head         <= out_rd_addr + 1;

                            if (out_rd_addr + 1 >= tail) begin
                                out_state <= OUT_DONE;
                            end else begin
                                porta_addr  <= out_rd_addr + 1;
                                out_rd_addr <= out_rd_addr + 1;
                                out_state   <= OUT_READING;
                            end
                        end else begin
                            net_out_wr <= 1'b0;
                        end
                    end

                    OUT_DONE: begin
                        net_out_wr  <= 1'b0;
                        head        <= 8'd0;
                        tail        <= 8'd0;
                        word_count  <= 8'd0;
                        pkt_ready_r <= 1'b0;
                        cap_state   <= CAP_IDLE;
                        out_state   <= OUT_IDLE;
                    end
                endcase
            end else begin
                net_out_wr <= 1'b0;
                out_state  <= OUT_IDLE;
            end
        end
    end

endmodule
