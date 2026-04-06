`timescale 1ns/1ps
///////////////////////////////////////////////////////////////////////////////
// arm_cpu_wrapper.v — Lab 8 ARM CPU with PCI-style programming interface
//
// cpu_done detection — fourth and final approach:
//
// Problem history:
//   v1: pc_tN >= halt_addr  → fired immediately (T1/T2/T3 start at halt addr)
//   v2: pc stall detection  → timed out (initialisation issues)
//   v3: has_run + threshold → still premature (T0 reaches word 8 after 33
//       cycles of fetch, but the loop hasn't finished executing)
//
// Root cause: in a 5-stage pipelined processor, the PC races ~4-5 stages
// ahead of committed execution.  T0 fetches word 8 (the halt) after just
// 8*4=32 cycles, but the last STR hasn't completed yet.
//
// Correct approach: MINIMUM CYCLE COUNTER.
// The ARM program (14 iterations × 8 instructions, 4-thread interleave)
// requires a provable minimum number of cycles:
//   - T0 gets 1 slot every 4 cycles
//   - 14 iterations × 8 instr = 112 T0-slots minimum
//   - Pipeline depth = 5 stages → add 5×4 = 20 cycles for drain
//   - Total minimum ≈ 112×4 + 20 = 468 cycles
// We use 600 cycles as a safe guard with headroom.
//
// cpu_done = (min_cycles_elapsed) AND (T0 PC back at halt word 8)
//
// The min-cycles guard prevents premature assertion.  Once the minimum
// has elapsed, we check that T0 is at/past word 8 (halt instruction).
// T1/T2/T3 are always at their halt words after the first few cycles,
// so we only need to gate on the minimum cycle count + T0 halt.
///////////////////////////////////////////////////////////////////////////////

module arm_cpu_wrapper (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        prog_en,
    input  wire        prog_we,
    input  wire        dmem_sel,
    input  wire [31:0] prog_addr,
    input  wire [31:0] prog_wdata,
    output wire [31:0] prog_rdata,

    output wire        cpu_done,
    output wire        cpu_running
);

    wire rst     = ~rst_n;
    wire arm_rst = rst | prog_en;

    assign cpu_running = ~prog_en & ~rst;

    // ── IMEM / DMEM Port B ────────────────────────────────────────────
    wire        imem_web   = prog_we & ~dmem_sel;
    wire [8:0]  imem_addrb = prog_addr[10:2];
    wire [31:0] imem_dinb  = prog_wdata;

    wire        dmem_web   = prog_we & dmem_sel;
    wire [7:0]  dmem_addrb = prog_addr[9:2];
    wire [31:0] dmem_dinb  = prog_wdata;

    // ── PC outputs ───────────────────────────────────────────────────
    wire [8:0] pc_t0, pc_t1, pc_t2, pc_t3;

    // ── Minimum cycle counter ────────────────────────────────────────
    // Counts cycles since ARM was released from reset.
    // 600 cycles is a safe lower bound for the 14-iteration add-1 program.
    // Adjust upward if running a longer program.
    localparam MIN_CYCLES = 10'd600;

    reg [9:0]  cycle_count;
    reg        min_elapsed;

    always @(posedge clk or posedge arm_rst) begin
        if (arm_rst) begin
            cycle_count <= 10'd0;
            min_elapsed <= 1'b0;
        end else begin
            if (cycle_count < MIN_CYCLES)
                cycle_count <= cycle_count + 1'b1;
            else
                min_elapsed <= 1'b1;
        end
    end

    // ── cpu_done: minimum elapsed AND T0 at halt (word 8) ────────────
    // Registered to avoid timing violation (Lab 6 pattern).
    // Gated by ~prog_en so it deasserts when prog mode re-enables.
    reg cpu_done_reg;
    always @(posedge clk or posedge rst) begin
        if (rst)
            cpu_done_reg <= 1'b0;
        else
            cpu_done_reg <= min_elapsed && (pc_t0 >= 9'd8);
    end
    assign cpu_done = cpu_done_reg & ~prog_en;

    // ── ARM Pipeline ─────────────────────────────────────────────────
    arm_pipeline arm_inst (
        .clk            (clk),
        .rst            (arm_rst),
        .ext_dmem_web   (dmem_web),
        .ext_dmem_addrb (dmem_addrb),
        .ext_dmem_dinb  (dmem_dinb),
        .ext_dmem_doutb (prog_rdata),
        .ext_imem_web   (imem_web),
        .ext_imem_addrb (imem_addrb),
        .ext_imem_dinb  (imem_dinb),
        .pc_t0          (pc_t0),
        .pc_t1          (pc_t1),
        .pc_t2          (pc_t2),
        .pc_t3          (pc_t3)
    );

endmodule
