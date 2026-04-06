// ==========================================
// Custom GPU ISA Definitions v2.0
// Target: NetFPGA (32-bit Instruction Width)
// ==========================================

// ------------------------------------------
// 1.1 Instruction Formats
// ------------------------------------------
// R-Type: [31:26] Opcode | [25:22] Rd | [21:18] Rs1 | [17:14] Rs2 | [13:2] Unused | [1:0] Flags
// I-Type: [31:26] Opcode | [25:22] Rd | [21:18] Rs_base | [17:0] Immediate (18-bit signed)
// ------------------------------------------

// ------------------------------------------
// 1.2 Supported Instructions (Opcodes)
// ------------------------------------------

// System / Control
`define OP_NOP       6'b000000
`define OP_READ_TID  6'b000001 // Rd <- Hardware Thread ID

// Memory Operations (64-bit aligned)
`define OP_LDI       6'b000010 // Rd <- Immediate (Sign extended)
`define OP_LD64      6'b000011 // Rd <- Mem[Rs_base + Imm]
`define OP_ST64      6'b000100 // Mem[Rs_base + Imm] <- Rd (Note: Rd acts as source here)

// Integer Vector Operations (4 x INT16)
`define OP_ADD_I16   6'b010000 // Rd <- Rs1 + Rs2
`define OP_SUB_I16   6'b010001 // Rd <- Rs1 - Rs2
`define OP_CMP_I16   6'b010010 // Compares Rs1 and Rs2, sets internal flags

// BFloat16 Vector & Tensor Operations (4 x BF16)
`define OP_BF_MUL    6'b100000 // Rd <- Rs1 * Rs2 (Standard parallel multiply)
`define OP_BF_MAC    6'b100001 // Tensor Unit: ACC <- ACC + (Rs1 * Rs2). (Uses 2 read ports!)
`define OP_RELU      6'b100010 // Rd <- max(0, Rs1)
`define OP_TF_START  6'b100011 // Triggers complex Tensor Unit pipeline
`define OP_TF_WAIT   6'b100100 // Stalls PC/Pipeline until Tensor Unit ready signal is high

// Branching
`define OP_BRANCH    6'b110000 // PC <- PC + Imm (Evaluates Flags set by CMP)

// ------------------------------------------
// 1.3 Register File & Hazard Rules (Comments for Verilog Implementation)
// ------------------------------------------
// - Read Ports: 2 (Rs1, Rs2)
// - Write Ports: 1 (Rd)
// - Writeback Timing: Cycle 3 (Fetch -> Decode/Read -> Execute/Writeback)
// - Hazard Behavior: No data forwarding. Compiler must insert NOPs, or 
//   Decode stage must stall if reading a register currently in the Execute stage.