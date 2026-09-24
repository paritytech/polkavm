# A store and a load through `x0` at a non-zero offset: an absolute access below 2 KiB.
#
# LLVM emits these at -O3 on a path it has proven dereferences null plus a field offset
# (midnight-proofs `multi_prepare`, inlined into a JAM guest). The PVM never maps the low
# 64 KiB, so each can only trap.
#
# Producer (the .o beside this file is its output; regenerate, never hand-edit):
#   clang --target=riscv64 -march=rv64imac -c test-data/x0-access.s -o test-data/x0-access.o
    .text
    .globl f
f:
    sd a0, 8(zero)
    ld a1, 16(zero)
    ret
