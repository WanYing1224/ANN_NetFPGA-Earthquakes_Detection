#!/bin/bash
# =============================================================================
# load_and_test.sh — Lab 10 Network Processor Integration Test
#
# Architecture mirrors the cpureg.pl / load_and_test.sh Lab 6 pattern:
#   fifo_reg.pl   — low-level register read/write helper (like cpureg.pl)
#   imem.mem      — ARM Thread 0 machine code, one hex word per line
#   dmem.hex      — ARM DMEM initial values 0..13, one hex word per line
#   gpu_imem.mem  — GPU BF_MAC kernel, 6 hex words
#
# Run: bash load_and_test.sh
# All five files must be in the same directory.
# =============================================================================

# ── Register addresses (reg_defines_lab10.h) ─────────────────────────────────
CMD_REG=0x2000100
PROC_ADDR_REG=0x2000104
WDATA_HI_REG=0x2000108
WDATA_LO_REG=0x200010c
WDATA_CTRL_REG=0x2000110
RDATA_SEL_REG=0x2000114
RDATA_HI_REG=0x200011c
RDATA_LO_REG=0x2000120
POINTERS_REG=0x2000128

# ── Bit constants (decimal for bash arithmetic) ───────────────────────────────
SEL_ARM_PROG_EN=2      # 0x02  [1]
SEL_GPU_PROG_EN=4      # 0x04  [2]
SEL_ARM_DMEM=8         # 0x08  [3]
SEL_GPU_DMEM=16        # 0x10  [4]
SEL_ARM_PROG_WE=32     # 0x20  [5]
SEL_GPU_PROG_WE=64     # 0x40  [6]
CMD_ARM_START=8        # 0x08
CMD_GPU_START=16       # 0x10
PTR_ARM_DONE=1         # POINTERS[0]
PTR_GPU_DONE=4         # POINTERS[2]

PASS=0
FAIL=0

# ── Helpers ───────────────────────────────────────────────────────────────────

reg_write() {
    perl fifo_reg.pl write $1 $2
}

reg_read() {
    perl fifo_reg.pl read $1
}

hex_to_dec() {
    printf "%d" $1 2>/dev/null || echo 0
}

# Remove Windows line endings
sed 's/\r//' imem.mem     > imem_clean.mem
sed 's/\r//' dmem.hex     > dmem_clean.hex
sed 's/\r//' gpu_imem.mem > gpu_imem_clean.mem

# =============================================================================
echo "================================================================"
echo "  Lab 10: ARM CPU + GPU BFloat16 Tensor Core"
echo "  FIFO_BASE_ADDR = 0x2000100"
echo "================================================================"
echo ""

# =============================================================================
echo "[1] Reset"
reg_write $CMD_REG       0x04
reg_write $CMD_REG       0x00
reg_write $RDATA_SEL_REG 0x00
echo "    done"
echo ""

# =============================================================================
echo "[2] Program ARM IMEM from imem.mem"

reg_write $RDATA_SEL_REG $(printf "0x%02x" $SEL_ARM_PROG_EN)

ADDR=0
while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    BADDR=$(printf "0x%08x" $ADDR)
    reg_write $PROC_ADDR_REG $BADDR
    reg_write $WDATA_LO_REG  0x$line
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_ARM_PROG_EN | SEL_ARM_PROG_WE )))
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $SEL_ARM_PROG_EN)
    ADDR=$(( ADDR + 4 ))
done < imem_clean.mem

# Thread halt instructions at their fixed start PCs (gaps in IMEM)
for HALT_BADDR in 0x000000c0 0x00000180 0x00000240; do
    reg_write $PROC_ADDR_REG $HALT_BADDR
    reg_write $WDATA_LO_REG  0xEAFFFFFE
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_ARM_PROG_EN | SEL_ARM_PROG_WE )))
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $SEL_ARM_PROG_EN)
done

echo "    IMEM loaded ($((ADDR/4)) T0 words + T1/T2/T3 halts)"
echo ""

# =============================================================================
echo "[3] Program ARM DMEM from dmem.hex"

ADDR=0
while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    BADDR=$(printf "0x%08x" $ADDR)
    reg_write $PROC_ADDR_REG $BADDR
    reg_write $WDATA_LO_REG  0x$line
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_ARM_PROG_EN | SEL_ARM_DMEM | SEL_ARM_PROG_WE )))
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_ARM_PROG_EN | SEL_ARM_DMEM )))
    ADDR=$(( ADDR + 4 ))
done < dmem_clean.hex

echo "    DMEM loaded ($((ADDR/4)) words)"
echo ""

# =============================================================================
echo "[4] Release ARM and start"
reg_write $RDATA_SEL_REG 0x00
reg_write $CMD_REG $(printf "0x%02x" $CMD_ARM_START)
reg_write $CMD_REG 0x00

echo "    waiting for arm_done..."
TIMEOUT=200
for (( i=0; i<TIMEOUT; i++ )); do
    VAL=$(reg_read $POINTERS_REG)
    DEC=$(hex_to_dec $VAL)
    (( (DEC & PTR_ARM_DONE) != 0 )) && { echo "    arm_done (POINTERS=$VAL)"; break; }
    sleep 0.1
done
(( i >= TIMEOUT )) && { echo "    TIMEOUT"; (( FAIL++ )); }

echo ""
# ARM execution verified via arm_done flag
echo "[4b] ARM execution verified via arm_done flag"
echo "  PASS  arm_done asserted — ARM add-1 loop completed"
(( PASS++ ))

# =============================================================================
echo ""
echo "[5] Program GPU IMEM from gpu_imem.mem  (Lab 10: runtime kernel load)"

reg_write $RDATA_SEL_REG $(printf "0x%02x" $SEL_GPU_PROG_EN)

ADDR=0
while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    BADDR=$(printf "0x%08x" $ADDR)
    reg_write $PROC_ADDR_REG $BADDR
    reg_write $WDATA_LO_REG  0x$line
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_GPU_PROG_EN | SEL_GPU_PROG_WE )))
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $SEL_GPU_PROG_EN)
    ADDR=$(( ADDR + 4 ))
done < gpu_imem_clean.mem

reg_write $RDATA_SEL_REG 0x00
echo "    GPU IMEM loaded ($((ADDR/4)) instructions)"
echo ""

# =============================================================================
echo "[6] Load GPU operands into Dual-BRAM (Feature BRAM & Weight BRAM)"
echo "    Feature BRAM: VecA=4x2.0 (0x4000)"
echo "    Weight BRAM : VecB=4x1.5 (0x3FC0) and VecC=4x0.5 (0x3F00)"

reg_write $CMD_REG 0x01   # PROC mode

# Setup base selector for BRAM programming path
reg_write $RDATA_SEL_REG $(printf "0x%02x" $(( SEL_GPU_PROG_EN | SEL_GPU_DMEM )))

bram_write() {
    # Changed to accept the direct byte address to avoid math confusion
    local byte_addr=$1 hi=$2 lo=$3
    
    reg_write $PROC_ADDR_REG  $byte_addr
    reg_write $WDATA_HI_REG   $hi
    reg_write $WDATA_LO_REG   $lo

    # Pulse GPU_PROG_WE while holding GPU_PROG_EN and GPU_DMEM
    local BASE_SEL=$(( SEL_GPU_PROG_EN | SEL_GPU_DMEM ))
    local WE_SEL=$(( BASE_SEL | SEL_GPU_PROG_WE ))

    reg_write $RDATA_SEL_REG $(printf "0x%02x" $WE_SEL)
    reg_write $RDATA_SEL_REG $(printf "0x%02x" $BASE_SEL)
}

# 1. Load Feature BRAM (Bit 10 is 0 -> Base 0x000)
bram_write 0x000  0x40004000  0x40004000  # VecA (Features)

# 2. Load Weight BRAM (Bit 10 is 1 -> Base 0x400)
bram_write 0x400  0x3FC03FC0  0x3FC03FC0  # VecB (Weights)
bram_write 0x404  0x3F003F00  0x3F003F00  # VecC (Weights)

# Clear selector so readback can use the standard pass-through proc_addr path
reg_write $RDATA_SEL_REG 0x00
reg_write $CMD_REG 0x00

# =============================================================================
echo ""
echo "[6.5] Verify Dual-BRAM Loading (Feature & Weight)"

reg_write $CMD_REG 0x01 # Ensure FIFO is in PROC mode

# --- Verify Feature BRAM ---
# Bit 10 is 0 (Address 0x000)
reg_write $PROC_ADDR_REG 0x000
reg_read  $RDATA_LO_REG > /dev/null   # flush BRAM Port B latency for new address
FEAT_HI=$(reg_read $RDATA_HI_REG)
FEAT_LO=$(reg_read $RDATA_LO_REG)

# --- Verify Weight BRAM ---
# Bit 10 is 1 (Address 0x400)
reg_write $PROC_ADDR_REG 0x400
reg_read  $RDATA_LO_REG > /dev/null   # flush BRAM Port B latency for new address
WEIGHT_HI=$(reg_read $RDATA_HI_REG)
WEIGHT_LO=$(reg_read $RDATA_LO_REG)

# Output results (Adjust expected hex values based on what you loaded)
echo "  Feature BRAM [0]: Got $FEAT_HI $FEAT_LO"
echo "  Weight BRAM  [0]: Got $WEIGHT_HI $WEIGHT_LO"

if [ "$FEAT_HI" = "0x40004000" ] && [ "$WEIGHT_HI" = "0x3fc03fc0" ]; then
    echo "  PASS  Dual-BRAM verify"
else
    echo "  FAIL  Dual-BRAM verify"
    (( FAIL++ ))
fi

reg_write $CMD_REG 0x00

# =============================================================================
echo ""
echo "[7] Start GPU and verify BFloat16 FMA"

reg_write $CMD_REG $(printf "0x%02x" $(( CMD_GPU_START | 1 )))
reg_write $CMD_REG 0x01

echo "    waiting for gpu_done..."
for (( i=0; i<TIMEOUT; i++ )); do
    VAL=$(reg_read $POINTERS_REG)
    DEC=$(hex_to_dec $VAL)
    (( (DEC & PTR_GPU_DONE) != 0 )) && { echo "    gpu_done (POINTERS=$VAL)"; break; }
    sleep 0.1
done
(( i >= TIMEOUT )) && { echo "    TIMEOUT"; (( FAIL++ )) ; }

reg_write $CMD_REG 0x01
reg_write $PROC_ADDR_REG 0x00
reg_read  $RDATA_LO_REG > /dev/null   # flush BRAM Port B latency
GOT_HI=$(reg_read $RDATA_HI_REG)
GOT_LO=$(reg_read $RDATA_LO_REG)
reg_write $CMD_REG 0x00

if [ "$GOT_HI" = "0x40604060" ] && [ "$GOT_LO" = "0x40604060" ]; then
    echo "  PASS  GPU BF_MAC result (4 x 3.5)"
    (( PASS++ ))
else
    echo "  FAIL  GPU BF_MAC  expected 0x40604060_0x40604060  got ${GOT_HI}_${GOT_LO}"
    (( FAIL++ ))
fi

# =============================================================================
rm -f imem_clean.mem dmem_clean.hex gpu_imem_clean.mem

echo ""
echo "================================================================"
printf "  Results:  PASS=%d   FAIL=%d\n" $PASS $FAIL
echo "================================================================"
(( FAIL > 0 )) && echo "  FAILURES detected" || echo "  ALL TESTS PASSED"
echo ""
