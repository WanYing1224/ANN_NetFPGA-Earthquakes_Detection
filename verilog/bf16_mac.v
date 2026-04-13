`timescale 1ns/1ps
// ============================================================
// bf16_mac.v — BFloat16 Fused Multiply-Add (Fully Pipelined)
//
// Optimized with 3 pipeline stages:
//   Stage 1: Multiplier (MULT18X18S built-in registers) and input alignment
//   Stage 2: Exponent difference, alignment shift (Barrel Shifter) 
//   Stage 3: Mantissa addition (Mantissa Add/Sub) and normalization
// ============================================================
module bf16_mac (
    input  wire        clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire [15:0] c,
    output wire [15:0] z
);

    // ── Unpack BF16 fields ────────────────────────────────────────
    wire        sign_a = a[15];
    wire [7:0]  exp_a  = a[14:7];
    wire [6:0]  mant_a = a[6:0];
    wire        sign_b = b[15];
    wire [7:0]  exp_b  = b[14:7];
    wire [6:0]  mant_b = b[6:0];
    wire        sign_c = c[15];
    wire [7:0]  exp_c  = c[14:7];
    wire [6:0]  mant_c = c[6:0];

    // ────────────────────────────────────────────────────────────
    // 1. MANTISSA MULTIPLY via MULT18X18S
    // ────────────────────────────────────────────────────────────
    wire [7:0]  mant_a_full = {1'b1, mant_a};
    wire [7:0]  mant_b_full = {1'b1, mant_b};
    wire [35:0] mult_p;

    MULT18X18S mult_inst (
        .C  (clk),
        .CE (1'b1),
        .R  (1'b0),
        .A  ({10'b0, mant_a_full}),
        .B  ({10'b0, mant_b_full}),
        .P  (mult_p)
    );

    // ============================================================
    // 🚧 PIPELINE STAGE 1: Aligning with Multiplier Latency
    // ============================================================
    reg        sign_a_r1, sign_b_r1, sign_c_r1;
    reg [7:0]  exp_a_r1, exp_b_r1, exp_c_r1;
    reg [6:0]  mant_c_r1;
    reg [15:0] a_r1, b_r1, c_r1;

    always @(posedge clk) begin
        sign_a_r1 <= sign_a; sign_b_r1 <= sign_b; sign_c_r1 <= sign_c;
        exp_a_r1  <= exp_a;  exp_b_r1  <= exp_b;  exp_c_r1  <= exp_c;
        mant_c_r1 <= mant_c;
        a_r1      <= a;      b_r1      <= b;      c_r1      <= c;
    end

    wire [15:0] mul_mant_temp = mult_p[15:0];
    wire        mul_norm      = mul_mant_temp[15];
    wire [6:0]  mul_mant      = mul_norm ? mul_mant_temp[14:8] : mul_mant_temp[13:7];
    wire        mul_sign      = sign_a_r1 ^ sign_b_r1;

    // ────────────────────────────────────────────────────────────
    // 2. EXPONENT COMPUTE (using _r1 registers)
    // ────────────────────────────────────────────────────────────
    wire [8:0] carry_a;
    wire [7:0] half_a, sum_ab;
    wire [8:0] carry_b;
    wire [7:0] half_b, sub127;
    wire [7:0] inv127 = 8'h80;
    wire [8:0] carry_c;
    wire [7:0] half_c, mul_exp;

    assign carry_a[0] = 1'b0;
    assign carry_b[0] = 1'b1;
    assign carry_c[0] = mul_norm;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : exp_compute
            LUT2 #(.INIT(4'h6)) ha_a (.I0(exp_a_r1[i]), .I1(exp_b_r1[i]), .O(half_a[i]));
            MUXCY mca (.O(carry_a[i+1]), .CI(carry_a[i]), .DI(exp_a_r1[i]), .S(half_a[i]));
            XORCY xca (.O(sum_ab[i]),    .CI(carry_a[i]), .LI(half_a[i]));
        end
        for (i = 0; i < 8; i = i + 1) begin : sub127_chain
            LUT2 #(.INIT(4'h6)) ha_b (.I0(sum_ab[i]), .I1(inv127[i]), .O(half_b[i]));
            MUXCY mcb (.O(carry_b[i+1]), .CI(carry_b[i]), .DI(sum_ab[i]), .S(half_b[i]));
            XORCY xcb (.O(sub127[i]),    .CI(carry_b[i]), .LI(half_b[i]));
        end
        for (i = 0; i < 8; i = i + 1) begin : norm_add_chain
            MUXCY mcc (.O(carry_c[i+1]), .CI(carry_c[i]), .DI(1'b0), .S(sub127[i]));
            XORCY xcc (.O(mul_exp[i]),   .CI(carry_c[i]), .LI(sub127[i]));
        end
    endgenerate

    // ────────────────────────────────────────────────────────────
    // 3. MAGNITUDE COMPARATOR
    // ────────────────────────────────────────────────────────────
    wire [8:0] carry_cmp;
    wire [7:0] inv_exp_c, half_cmp;
    assign carry_cmp[0] = 1'b0;

    generate
        for (i = 0; i < 8; i = i + 1) begin : comparator
            LUT2 #(.INIT(4'h5)) inv_c (.I0(exp_c_r1[i]), .I1(1'b0), .O(inv_exp_c[i]));
            LUT2 #(.INIT(4'h6)) hcmp  (.I0(mul_exp[i]), .I1(inv_exp_c[i]), .O(half_cmp[i]));
            MUXCY mccmp (.O(carry_cmp[i+1]), .CI(carry_cmp[i]), .DI(mul_exp[i]), .S(half_cmp[i]));
        end
    endgenerate

    wire mul_is_bigger = carry_cmp[8];

    wire [15:0] mul_res = (a_r1 == 16'd0 || b_r1 == 16'd0) ? 16'd0 : {mul_sign, mul_exp, mul_mant};
    wire [7:0] m_mul = (mul_res == 16'd0) ? 8'd0 : {1'b1, mul_mant};
    wire [7:0] m_c   = (c_r1 == 16'd0)    ? 8'd0 : {1'b1, mant_c_r1};
    wire [7:0] final_exp = mul_is_bigger ? mul_exp : exp_c_r1;

    // ────────────────────────────────────────────────────────────
    // 4. EXPONENT DIFFERENCE
    // ────────────────────────────────────────────────────────────
    wire [7:0] sub1_b_inv, sub2_b_inv, sub1_half, sub2_half, sub1_out, sub2_out;
    wire [8:0] sub1_carry, sub2_carry;
    assign sub1_carry[0] = 1'b1;
    assign sub2_carry[0] = 1'b1;

    generate
        for (i = 0; i < 8; i = i + 1) begin : exp_diff_sub
            LUT2 #(.INIT(4'h5)) inv1 (.I0(exp_c_r1[i]),   .I1(1'b0), .O(sub1_b_inv[i]));
            LUT2 #(.INIT(4'h5)) inv2 (.I0(mul_exp[i]), .I1(1'b0), .O(sub2_b_inv[i]));

            LUT2 #(.INIT(4'h6)) hs1 (.I0(mul_exp[i]), .I1(sub1_b_inv[i]), .O(sub1_half[i]));
            MUXCY mc1 (.O(sub1_carry[i+1]), .CI(sub1_carry[i]), .DI(mul_exp[i]), .S(sub1_half[i]));
            XORCY xc1 (.O(sub1_out[i]),     .CI(sub1_carry[i]), .LI(sub1_half[i]));

            LUT2 #(.INIT(4'h6)) hs2 (.I0(exp_c_r1[i]),   .I1(sub2_b_inv[i]), .O(sub2_half[i]));
            MUXCY mc2 (.O(sub2_carry[i+1]), .CI(sub2_carry[i]), .DI(exp_c_r1[i]),   .S(sub2_half[i]));
            XORCY xc2 (.O(sub2_out[i]),     .CI(sub2_carry[i]), .LI(sub2_half[i]));
        end
    endgenerate

    wire [7:0] exp_diff = mul_is_bigger ? sub1_out : sub2_out;

    // ────────────────────────────────────────────────────────────
    // 5. ALIGNMENT BARREL SHIFTER
    // ────────────────────────────────────────────────────────────
    wire [7:0] sh_mul_s0, sh_mul_s1, sh_mul_s2;
    wire [7:0] sh_c_s0,   sh_c_s1,   sh_c_s2;

    generate
        for (i = 0; i < 8; i = i + 1) begin : shr_mul_s0
            wire in_hi = (i < 7) ? m_mul[i+1] : 1'b0;
            MUXF5 mux0 (.O(sh_mul_s0[i]), .I0(m_mul[i]), .I1(in_hi), .S(exp_diff[0]));
        end
        for (i = 0; i < 8; i = i + 1) begin : shr_mul_s1
            wire in_hi2 = (i < 6) ? sh_mul_s0[i+2] : 1'b0;
            MUXF5 mux1 (.O(sh_mul_s1[i]), .I0(sh_mul_s0[i]), .I1(in_hi2), .S(exp_diff[1]));
        end
        for (i = 0; i < 8; i = i + 1) begin : shr_mul_s2
            wire in_hi4 = (i < 4) ? sh_mul_s1[i+4] : 1'b0;
            MUXF5 mux2 (.O(sh_mul_s2[i]), .I0(sh_mul_s1[i]), .I1(in_hi4), .S(exp_diff[2]));
        end
        for (i = 0; i < 8; i = i + 1) begin : shr_c_s0
            wire in_hi = (i < 7) ? m_c[i+1] : 1'b0;
            MUXF5 mux0 (.O(sh_c_s0[i]), .I0(m_c[i]), .I1(in_hi), .S(exp_diff[0]));
        end
        for (i = 0; i < 8; i = i + 1) begin : shr_c_s1
            wire in_hi2 = (i < 6) ? sh_c_s0[i+2] : 1'b0;
            MUXF5 mux1 (.O(sh_c_s1[i]), .I0(sh_c_s0[i]), .I1(in_hi2), .S(exp_diff[1]));
        end
        for (i = 0; i < 8; i = i + 1) begin : shr_c_s2
            wire in_hi4 = (i < 4) ? sh_c_s1[i+4] : 1'b0;
            MUXF5 mux2 (.O(sh_c_s2[i]), .I0(sh_c_s1[i]), .I1(in_hi4), .S(exp_diff[2]));
        end
    endgenerate

    wire [7:0] aligned_mul_m = mul_is_bigger ? m_mul    : sh_mul_s2;
    wire [7:0] aligned_c_m   = mul_is_bigger ? sh_c_s2  : m_c;

    wire signs_match = (mul_sign == sign_c_r1);
    wire final_sign  = mul_is_bigger ? mul_sign : sign_c_r1;

    wire [7:0] op_a = (signs_match || mul_is_bigger) ? aligned_mul_m : aligned_c_m;
    wire [7:0] op_b_raw = (signs_match || mul_is_bigger) ? aligned_c_m : aligned_mul_m;
    wire do_sub = ~signs_match;

    // ============================================================
    // 🚧 PIPELINE STAGE 2: Buffering before Mantissa Addition
    // ============================================================
    reg [7:0] op_a_r2, op_b_raw_r2;
    reg       do_sub_r2, final_sign_r2;
    reg [7:0] final_exp_r2;
    reg [15:0] mul_res_r2, c_r2;

    always @(posedge clk) begin
        op_a_r2       <= op_a;
        op_b_raw_r2   <= op_b_raw;
        do_sub_r2     <= do_sub;
        final_sign_r2 <= final_sign;
        final_exp_r2  <= final_exp;
        mul_res_r2    <= mul_res;
        c_r2          <= c_r1;
    end

    // ────────────────────────────────────────────────────────────
    // 6. MANTISSA ADD/SUBTRACT (using _r2 registers)
    // ────────────────────────────────────────────────────────────
    wire [7:0] op_b_inv;
    wire [8:0] carry_sum;
    wire [7:0] half_sum;
    wire [8:0] sum_mant; 

    generate
        for (i = 0; i < 8; i = i + 1) begin : op_b_invert
            LUT2 #(.INIT(4'h6)) inv_b (.I0(op_b_raw_r2[i]), .I1(do_sub_r2), .O(op_b_inv[i]));
        end
    endgenerate

    assign carry_sum[0] = do_sub_r2;

    generate
        for (i = 0; i < 8; i = i + 1) begin : mantissa_add
            LUT2 #(.INIT(4'h6)) hs (.I0(op_a_r2[i]), .I1(op_b_inv[i]), .O(half_sum[i]));
            MUXCY mcs (.O(carry_sum[i+1]), .CI(carry_sum[i]), .DI(op_a_r2[i]), .S(half_sum[i]));
            XORCY xcs (.O(sum_mant[i]),    .CI(carry_sum[i]), .LI(half_sum[i]));
        end
    endgenerate

    assign sum_mant[8] = carry_sum[8]; 

    // ────────────────────────────────────────────────────────────
    // 7. NORMALIZE EXPONENT 
    // ────────────────────────────────────────────────────────────
    wire        add_norm = sum_mant[8];
    wire [8:0]  carry_ne;
    wire [7:0]  norm_exp;

    assign carry_ne[0] = add_norm;

    generate
        for (i = 0; i < 8; i = i + 1) begin : norm_exp_chain
            MUXCY mcne (.O(carry_ne[i+1]), .CI(carry_ne[i]),
                        .DI(1'b0), .S(final_exp_r2[i]));
            XORCY xcne (.O(norm_exp[i]),   .CI(carry_ne[i]),
                        .LI(final_exp_r2[i]));
        end
    endgenerate

    wire [6:0] norm_mant = add_norm ? sum_mant[7:1] : sum_mant[6:0];

    // ============================================================
    // PIPELINE STAGE 3 Preparation: Final output will be captured 
    // by registers within tensor_unit.v
    // ============================================================
    assign z = (mul_res_r2 == 16'd0 && c_r2 == 16'd0) ? 16'd0
             : (mul_res_r2 == 16'd0)                  ? c_r2
             : (c_r2 == 16'd0)                        ? mul_res_r2
             : {final_sign_r2, norm_exp, norm_mant};

endmodule
