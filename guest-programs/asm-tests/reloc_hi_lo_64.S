.global get_string

get_string:
    slli a0, a0, 2
    lui a1, %hi(magic_table)
    addi a1, a1, %lo(magic_table)
    mv a2, a1
    add a1, a1, a0
    lw a0, 0(a1)
    add a0, a0, a2
    ret

.pushsection .metadata,"",@progbits
_get_string_name:
    .asciz "get_string"
_get_string_name_end:

_metadata:
    .byte 1
    .word 0
    .word _get_string_name_end - _get_string_name - 1
    .quad _get_string_name
    .byte 1
    .byte 1
.popsection

.pushsection .polkavm_exports,"R",@note
    .byte 1
    .quad _metadata
    .quad get_string
.popsection

.pushsection .rodata.secrets
magic0: .word 0xA1010101
magic1: .word 0xB2020202
magic2: .word 0xC3030303
.popsection

.pushsection .rodata.table
magic_table:
    .word magic0 - magic_table
    .word magic1 - magic_table
    .word magic2 - magic_table
.popsection

