	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 26, 0	sdk_version 26, 1
	.globl	_dense_relu                     ; -- Begin function dense_relu
	.p2align	2
_dense_relu:                            ; @dense_relu
	.cfi_startproc
; %bb.0:
	stp	d15, d14, [sp, #-64]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	.cfi_offset b14, -56
	.cfi_offset b15, -64
	mov	x8, #0                          ; =0x0
	ldp	s0, s1, [x0]
	ldp	s2, s3, [x0, #8]
	ldp	s4, s5, [x0, #16]
	ldp	s6, s7, [x0, #24]
	ldp	s16, s17, [x0, #32]
	ldp	s18, s19, [x0, #40]
	ldp	s20, s21, [x0, #48]
	ldp	s22, s23, [x0, #56]
	ldp	s24, s25, [x0, #64]
	ldp	s26, s27, [x0, #72]
	ldp	d28, d29, [x0, #80]
	ldp	d30, d31, [x0, #96]
	ldp	d8, d9, [x0, #112]
	add	x9, x1, #60
	movi	d10, #0000000000000000
LBB0_1:                                 ; =>This Inner Loop Header: Depth=1
	ldp	s11, s12, [x9, #-60]
	fmul	s11, s0, s11
	fadd	s11, s11, s10
	fmul	s12, s1, s12
	ldp	s13, s14, [x9, #-52]
	fmul	s13, s2, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s3, s14
	ldp	s13, s14, [x9, #-44]
	fmul	s13, s4, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s5, s14
	ldp	s13, s14, [x9, #-36]
	fmul	s13, s6, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s7, s14
	ldp	s13, s14, [x9, #-28]
	fmul	s13, s16, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s17, s14
	ldp	s13, s14, [x9, #-20]
	fmul	s13, s18, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s19, s14
	ldp	s13, s14, [x9, #-12]
	fmul	s13, s20, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s21, s14
	ldp	s13, s14, [x9, #-4]
	fmul	s13, s22, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s23, s14
	ldp	s13, s14, [x9, #4]
	fmul	s13, s24, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s25, s14
	ldp	s13, s14, [x9, #12]
	fmul	s13, s26, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	fmul	s12, s27, s14
	fadd	s11, s11, s12
	ldur	d12, [x9, #20]
	fmul.2s	v12, v28, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #28]
	fmul.2s	v12, v29, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #36]
	fmul.2s	v12, v30, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #44]
	fmul.2s	v12, v31, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	ldur	d13, [x9, #52]
	fadd	s11, s11, s12
	fmul.2s	v12, v8, v13
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #60]
	fmul.2s	v12, v9, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldr	s12, [x2, x8]
	fadd	s11, s11, s12
	fmaxnm	s11, s11, s10
	str	s11, [x3, x8]
	add	x9, x9, #128
	add	x8, x8, #4
	cmp	x8, #128
	b.ne	LBB0_1
; %bb.2:
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #64             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
