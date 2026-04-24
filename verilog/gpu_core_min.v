`timescale 1ns/1ps
// ============================================================
// gpu_core_min.v — Minimal Programmable GPU Core
// ============================================================
`include "isa_defines.vh"

module gpu_core_min (
    input  wire        clk,
    input  wire        rst,
    
    input  wire        start,
    output reg         done,
    output reg         running,

    // Programming interface
    input  wire        gpu_prog_en,
    input  wire        gpu_prog_we,
    input  wire        gpu_dmem_sel,
    input  wire [31:0] gpu_prog_addr,
    input  wire [31:0] gpu_prog_wdata,

    // Shared FIFO BRAM interface
    output wire [7:0]  bram_addr,
    output wire [71:0] bram_wdata,
    output wire        bram_we,
    input  wire [71:0] bram_rdata,
    
    // Shared Weight BRAM interface
    output wire [7:0]  weight_addr,
    input  wire [71:0] weight_rdata,

    // Shared Bias ROM interface (NEW)
    output wire [7:0]  bias_addr,
    input  wire [71:0] bias_rdata
);

    // ── Programming mode reset ────────────────────────────────────
    wire core_rst = rst | gpu_prog_en;

    // ── IMEM Port B (host write) ──────────────────────────────────
    wire        imem_web   = gpu_prog_we & ~gpu_dmem_sel;
    wire [4:0]  imem_addrb = gpu_prog_addr[6:2];
    wire [31:0] imem_dinb  = gpu_prog_wdata;

    // ── FSM states ────────────────────────────────────────────────
    localparam S_IDLE       = 4'd0;
    localparam S_FETCH      = 4'd1;   // present pc to BRAM
    localparam S_FETCH_WAIT = 4'd2;   // wait one cycle for BRAM output
    localparam S_DECODE     = 4'd3;   // latch instruction, read regfile
    localparam S_EXEC       = 4'd4;
    localparam S_MEM        = 4'd5;
    localparam S_MEM2       = 4'd6;
    localparam S_WB         = 4'd7;
    localparam S_DONE       = 4'd8;

    reg [3:0] state;
    parameter PROG_LEN = 17; // Increased by 1 for the LDB64 instruction

    // ── PC ────────────────────────────────────────────────────────
    reg [4:0] pc;

    // ── Instruction Memory (programmable BRAM) ────────────────────
    wire [31:0] instr_from_mem;
    reg  [31:0] instr_reg;

    gpu_imem imem_inst (
        .clka  (clk),
        .addra (pc),
        .douta (instr_from_mem),
        .clkb  (clk),
        .web   (imem_web),
        .addrb (imem_addrb),
        .dinb  (imem_dinb)
    );

    // ── ISA field decode ──────────────────────
    wire [5:0]  opcode   = instr_reg[31:26];
    wire [3:0]  rd_4b    = instr_reg[25:22];
    wire [3:0]  rs1_4b   = instr_reg[21:18];
    wire [3:0]  rs2_4b   = instr_reg[17:14];
    wire [3:0]  rs3_4b   = instr_reg[13:10];

    wire [17:0] imm18    = instr_reg[17:0];
    wire [63:0] imm_sext = {{46{imm18[17]}}, imm18};

    wire [2:0] rd_addr  = rd_4b[2:0];
    wire [2:0] rs1_addr = rs1_4b[2:0];
    wire [2:0] rs2_addr = rs2_4b[2:0];
    wire [2:0] rs3_addr = rs3_4b[2:0];

    // ── Register File ─────────────────────────────────────────────
    reg         rf_we;
    reg  [2:0]  rf_wr_addr;
    reg  [63:0] rf_wr_data;
    wire [63:0] rs1_data, rs2_data, rs3_data, rd_src_data;

    gpu_regfile_min regfile_inst (
        .clk         (clk),
        .we          (rf_we),
        .rd_addr     (rf_wr_addr),
        .rd_data     (rf_wr_data),
        .rs1_addr    (rs1_addr),
        .rs2_addr    (rs2_addr),
        .rs3_addr    (rs3_addr),
        .rd_src_addr (rd_addr),
        .rs1_out     (rs1_data),
        .rs2_out     (rs2_data),
        .rs3_out     (rs3_data),
        .rd_src_out  (rd_src_data)
    );

    // ── Tensor Unit ───────────────────────────────────────────────
    reg         tensor_start;
    wire        tensor_busy, tensor_done;
    wire [63:0] tensor_result;

    tensor_unit tensor_inst (
        .clk    (clk),
        .rst    (core_rst),
        .start  (tensor_start),
        .rs1    (rs1_data),
        .rs2    (rs2_data),
        .rs3    (rs3_data),
        .busy   (tensor_busy),
        .done   (tensor_done),
        .result (tensor_result)
    );

    // ── ALU ───────────────────────────────────────────────────────
    wire signed [15:0] a0=rs1_data[15:0],  a1=rs1_data[31:16],
                       a2=rs1_data[47:32], a3=rs1_data[63:48];

    wire signed [15:0] b0=rs2_data[15:0],  b1=rs2_data[31:16],
                       b2=rs2_data[47:32], b3=rs2_data[63:48];

    reg [63:0] alu_result;
    always @(*) begin
        case (opcode)
            `OP_ADD_I16:  alu_result = {a3+b3, a2+b2, a1+b1, a0+b0};
            `OP_SUB_I16:  alu_result = {a3-b3, a2-b2, a1-b1, a0-b0};
            `OP_CMP_I16:  alu_result = {63'd0, (a0 < b0)};
            `OP_RELU:     alu_result = {(a3>0?a3:16'd0),(a2>0?a2:16'd0),
                                        (a1>0?a1:16'd0),(a0>0?a0:16'd0)};
            `OP_LDI:      alu_result = imm_sext;
            `OP_READ_TID: alu_result = 64'd0;
            `OP_LD64, `OP_LDW64, `OP_LDB64: alu_result = rs1_data + imm_sext; // ADDED LDB64
            `OP_ST64:     alu_result = rs1_data + imm_sext;
            default:      alu_result = 64'd0;
        endcase
    end

    reg cmp_flag;

    // ── LD/ST Unit ────────────────────────────────────────────────
    reg  [31:0] ldst_byte_addr;
    reg  [31:0] ldw_byte_addr;
    reg  [31:0] ldb_byte_addr; // NEW FOR BIAS

    reg  [63:0] ldst_st_data;
    reg         ldst_st_en;

    wire [7:0]  ld_bram_addr;
    wire [63:0] ld_data;
    
    wire [7:0]  rd_weight_addr; 
    wire [63:0] ldw_data;

    wire [7:0]  st_bram_addr;
    wire [71:0] st_bram_wdata;
    wire        st_bram_we;

    gpu_ldst ldst_inst (
        .ld_byte_addr  (ldst_byte_addr),
        .rd_bram_addr  (ld_bram_addr),
        .bram_rdata    (bram_rdata),
        .ld_data       (ld_data),
        
        .ldw_byte_addr (ldw_byte_addr),   
        .rd_weight_addr(rd_weight_addr),  
        .weight_rdata  (weight_rdata),    
        .ldw_data      (ldw_data),        

        .st_byte_addr  (ldst_byte_addr),
        .st_data       (ldst_st_data),
        .wr_bram_addr  (st_bram_addr),
        .wr_bram_wdata (st_bram_wdata),
        .wr_bram_we    (st_bram_we),
        .st_en         (ldst_st_en)
    );

    // Memory Routing
    assign bram_addr   = ldst_st_en ? st_bram_addr  : ld_bram_addr;
    assign bram_wdata  = st_bram_wdata;
    assign bram_we     = st_bram_we;
    assign weight_addr = rd_weight_addr;

    // Bias addressing directly integrated
    assign bias_addr   = {3'b0, ldb_byte_addr[7:3]};
    wire [63:0] ldb_data = bias_rdata[63:0];

    // ── Intermediate result register ──────────────────────────────
    reg [63:0] exec_result;

    // ── FSM ───────────────────────────────────────────────────────
    always @(posedge clk or posedge core_rst) begin
        if (core_rst) begin
            state          <= S_IDLE;
            pc             <= 5'd0;
            done           <= 1'b0;
            running        <= 1'b0;
            instr_reg      <= 32'd0;
            exec_result    <= 64'd0;
            ldst_byte_addr <= 32'd0;
            ldw_byte_addr  <= 32'd0;
            ldb_byte_addr  <= 32'd0; // NEW
            ldst_st_data   <= 64'd0;
            ldst_st_en     <= 1'b0;
            rf_we          <= 1'b0;
            tensor_start   <= 1'b0;
            cmp_flag       <= 1'b0;
        end else begin
            // Default: clear one-cycle pulses
            rf_we        <= 1'b0;
            ldst_st_en   <= 1'b0;
            tensor_start <= 1'b0;

            case (state)

                S_IDLE: begin
                    done    <= 1'b0;
                    running <= 1'b0;
                    pc      <= 5'd0;
                    if (start) begin
                        state   <= S_FETCH;
                        running <= 1'b1;
                    end
                end

                S_FETCH: begin
                    if (pc >= PROG_LEN) begin
                        state <= S_DONE;
                    end else begin
                        state <= S_FETCH_WAIT;
                    end
                end

                S_FETCH_WAIT: begin
                    instr_reg <= instr_from_mem;
                    state     <= S_DECODE;
                end

                S_DECODE: begin
                    state <= S_EXEC;
                end

                S_EXEC: begin
                    case (opcode)
                        `OP_BF_MAC: begin
                            tensor_start <= 1'b1;
                            state        <= S_MEM;
                        end
                        `OP_LD64: begin
                            ldst_byte_addr <= alu_result[31:0];
                            state          <= S_MEM;
                        end
                        `OP_LDW64: begin
                            ldw_byte_addr <= alu_result[31:0];
                            state         <= S_MEM;
                        end
                        `OP_LDB64: begin // NEW
                            ldb_byte_addr <= alu_result[31:0];
                            state         <= S_MEM;
                        end
                        `OP_ST64: begin
                            ldst_byte_addr <= alu_result[31:0];
                            ldst_st_data   <= rd_src_data;
                            ldst_st_en     <= 1'b1;
                            pc    <= pc + 5'd1;
                            state <= S_FETCH;
                        end
                        `OP_CMP_I16: begin
                            cmp_flag <= alu_result[0];
                            pc    <= pc + 5'd1;
                            state <= S_FETCH;
                        end
                        `OP_BRANCH: begin
                            if (cmp_flag)
                                pc <= pc + $signed(imm18[4:0]);
                            else
                                pc <= pc + 5'd1;
                            state <= S_FETCH;
                        end
                        `OP_NOP: begin
                            pc    <= pc + 5'd1;
                            state <= S_FETCH;
                        end
                        default: begin
                            exec_result <= alu_result;
                            state       <= S_WB;
                        end
                    endcase
                end

                S_MEM: begin
                    if (opcode == `OP_BF_MAC) begin
                        if (tensor_done) begin
                            exec_result <= tensor_result;
                            state       <= S_WB;
                        end
                    end else begin
                        state <= S_MEM2;
                    end
                end

                S_MEM2: begin
                    exec_result <= (opcode == `OP_LDW64) ? ldw_data : 
                                   (opcode == `OP_LDB64) ? ldb_data : ld_data; // ADDED LDB_DATA
                    state       <= S_WB;
                end

                S_WB: begin
                    if (rd_addr != 3'd0) begin
                        rf_we      <= 1'b1;
                        rf_wr_addr <= rd_addr;
                        rf_wr_data <= exec_result;
                    end
                    pc    <= pc + 5'd1;
                    state <= S_FETCH;
                end

                S_DONE: begin
                    done    <= 1'b1;
                    running <= 1'b0;
                    // Minimal board-side fix:
                    // a new start pulse should restart immediately from PC=0.
                    if (start) begin
                        done    <= 1'b0;
                        running <= 1'b1;
                        pc      <= 5'd0;
                        state   <= S_FETCH;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
