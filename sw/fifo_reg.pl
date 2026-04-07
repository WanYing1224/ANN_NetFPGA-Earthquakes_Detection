#!/usr/bin/perl -w
# =============================================================================
# fifo_reg.pl — Register helper for Lab 10 NetFPGA integration
#
# Usage (called by load_and_test.sh):
#   perl fifo_reg.pl write <addr> <val>
#   perl fifo_reg.pl read  <addr>
#
# Matches the cpureg.pl / load_and_test.sh pattern exactly.
# regread prints "Found net device: nf2c0" to stderr (discarded by backticks)
# and "Reg 0x... (dec):   0x... (dec)" to stdout (captured in @out).
# =============================================================================
use strict;

# ── Register addresses (from reg_defines_lab10.h) ─────────────────────────────
my $CMD_REG        = 0x2000100;
my $PROC_ADDR_REG  = 0x2000104;
my $WDATA_HI_REG   = 0x2000108;
my $WDATA_LO_REG   = 0x200010c;
my $WDATA_CTRL_REG = 0x2000110;
my $RDATA_SEL_REG  = 0x2000114;
my $STATUS_REG     = 0x2000118;
my $RDATA_HI_REG   = 0x200011c;
my $RDATA_LO_REG   = 0x2000120;
my $RDATA_CTRL_REG = 0x2000124;
my $POINTERS_REG   = 0x2000128;

# ── Low-level helpers ─────────────────────────────────────────────────────────

sub reg_write {
    my ($addr, $val) = @_;
    system(sprintf("regwrite 0x%08x 0x%08x", $addr, $val));
}

sub reg_read {
    my ($addr) = @_;
    my @out = `regread $addr`;
    for my $line (@out) {
        if ($line =~ /Reg\s+(0x[0-9a-fA-F]+)\s+\((\d+)\):\s+(0x[0-9a-fA-F]+)\s+\((\d+)\)/) {
            return $3;   # return hex string e.g. "0x00000007"
        }
    }
    return "0x00000000";
}

# ── Command dispatch ──────────────────────────────────────────────────────────

my $numargs = $#ARGV + 1;
if ($numargs < 2) {
    print "Usage: fifo_reg.pl <write|read> <addr> [val]\n";
    exit 1;
}

my $cmd  = $ARGV[0];
my $addr = $ARGV[1];

if ($cmd eq "write") {
    if ($numargs < 3) { die "Error: write requires address and value\n"; }
    reg_write(hex($addr), hex($ARGV[2]));

} elsif ($cmd eq "read") {
    print reg_read($addr), "\n";

} else {
    die "Unknown command: $cmd\n";
}
