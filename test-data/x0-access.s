# Every instruction the linker can meet with `x0` as its base or address, one exported function
# per row of `convert_instruction`'s handling. LLVM emits the load/store forms at -O3 on a path it
# has proven dereferences null plus a field offset (e.g. `sd a3, 0x8(zero)` in an inlined
# midnight-proofs `multi_prepare`).
#
# A relocatable object, as the linker accepts: every section sits at 0, so no row may be mistaken
# for an access into one. `norvc` makes row `n` the two words at offset 8n. No row reaches the last
# octet of the address space: the VM traps an access that ends at 2^32 even once its page is mapped.
#
# Producer (the .o beside this file is its output; regenerate, never hand-edit):
#   clang --target=riscv64 -march=rv64imac -c test-data/x0-access.s -o test-data/x0-access.o
# (clang 21.1.4; byte-identical across runs.)

.macro row name, insn:vararg
    .globl \name
\name:
    \insn
    ret

    .pushsection .metadata,"",@progbits
.L\name\()_symbol:
    .ascii "\name"
.L\name\()_symbol_end:
.L\name\()_metadata:
    .byte 1                                         # version
    .word 0                                         # flags
    .word .L\name\()_symbol_end - .L\name\()_symbol # symbol length
    .quad .L\name\()_symbol
    .byte 1                                         # input regs
    .byte 0                                         # output regs
    .popsection

    .pushsection .polkavm_exports,"R",@note
    .byte 1
    .quad .L\name\()_metadata
    .quad \name
    .popsection
.endm

    .option norvc
    .text
    row store_low,      sd a0, 8(zero)          # -> store to literal 0x8
    row load_low,       ld a1, 16(zero)         # -> load from literal 0x10
    row store_high,     sd a0, -16(zero)        # -> store to literal 0xfffffff0
    row load_high,      ld a1, -24(zero)        # -> load from literal 0xffffffe8
    row load_x0_low,    ld zero, 24(zero)       # -> load from literal 0x18 into a scratch register
    row load_x0_high,   ld zero, -32(zero)      # -> load from literal 0xffffffe0 into a scratch register
    row lr_w,           lr.w a1, (zero)         # atomic load at 0      -> trap
    row sc_w,           sc.w a2, a0, (zero)     # atomic store at 0     -> trap
    row amoadd_w,       amoadd.w a1, a0, (zero) # atomic op at 0        -> trap
    row jalr_low,       jalr ra, 32(zero)       # jump to literal 0x20  -> trap
    row trap_idiom,     sw zero, 0(zero)        # LLVM's trap idiom     -> trap, as ever
