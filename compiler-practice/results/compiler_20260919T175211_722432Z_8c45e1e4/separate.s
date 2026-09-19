	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 26, 0	sdk_version 26, 1
	.globl	_matmul_transpose               ; -- Begin function matmul_transpose
	.p2align	2
_matmul_transpose:                      ; @matmul_transpose
	.cfi_startproc
; %bb.0:
	stp	d13, d12, [sp, #-48]!           ; 16-byte Folded Spill
	stp	d11, d10, [sp, #16]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #32]               ; 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
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
	fmul	s12, s1, s12
	fadd	s11, s11, s10
	fadd	s11, s11, s12
	ldp	s12, s13, [x9, #-52]
	fmul	s12, s2, s12
	fmul	s13, s3, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-44]
	fmul	s12, s4, s12
	fmul	s13, s5, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-36]
	fmul	s12, s6, s12
	fmul	s13, s7, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-28]
	fmul	s12, s16, s12
	fmul	s13, s17, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-20]
	fmul	s12, s18, s12
	fmul	s13, s19, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-12]
	fmul	s12, s20, s12
	fmul	s13, s21, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #-4]
	fmul	s12, s22, s12
	fmul	s13, s23, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #4]
	fmul	s12, s24, s12
	fmul	s13, s25, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldp	s12, s13, [x9, #12]
	fmul	s12, s26, s12
	fmul	s13, s27, s13
	fadd	s11, s11, s12
	fadd	s11, s11, s13
	ldur	d12, [x9, #20]
	fmul.2s	v12, v28, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #28]
	fmul.2s	v12, v29, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	ldur	d13, [x9, #36]
	fadd	s11, s11, s12
	fmul.2s	v12, v30, v13
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #44]
	fmul.2s	v12, v31, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #52]
	fmul.2s	v12, v8, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	ldur	d12, [x9, #60]
	fmul.2s	v12, v9, v12
	fadd	s11, s11, s12
	mov	s12, v12[1]
	fadd	s11, s11, s12
	str	s11, [x2, x8]
	add	x9, x9, #128
	add	x8, x8, #4
	cmp	x8, #128
	b.ne	LBB0_1
; %bb.2:
	ldp	d9, d8, [sp, #32]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #16]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp], #48             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_add_bias                       ; -- Begin function add_bias
	.p2align	2
_add_bias:                              ; @add_bias
	.cfi_startproc
; %bb.0:
	ldp	q0, q1, [x0]
	ldp	q2, q3, [x1]
	fadd.4s	v0, v0, v2
	fadd.4s	v1, v1, v3
	stp	q0, q1, [x2]
	ldp	q0, q1, [x0, #32]
	ldp	q2, q3, [x1, #32]
	fadd.4s	v0, v0, v2
	fadd.4s	v1, v1, v3
	stp	q0, q1, [x2, #32]
	ldp	q0, q1, [x0, #64]
	ldp	q2, q3, [x1, #64]
	fadd.4s	v0, v0, v2
	fadd.4s	v1, v1, v3
	stp	q0, q1, [x2, #64]
	ldp	q0, q1, [x0, #96]
	ldp	q2, q3, [x1, #96]
	fadd.4s	v0, v0, v2
	fadd.4s	v1, v1, v3
	stp	q0, q1, [x2, #96]
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_relu                           ; -- Begin function relu
	.p2align	2
_relu:                                  ; @relu
	.cfi_startproc
; %bb.0:
	movi.2d	v0, #0000000000000000
	ldp	q1, q2, [x0]
	fmaxnm.4s	v1, v1, v0
	fmaxnm.4s	v2, v2, v0
	stp	q1, q2, [x1]
	ldp	q1, q2, [x0, #32]
	fmaxnm.4s	v1, v1, v0
	fmaxnm.4s	v2, v2, v0
	stp	q1, q2, [x1, #32]
	ldp	q1, q2, [x0, #64]
	fmaxnm.4s	v1, v1, v0
	fmaxnm.4s	v2, v2, v0
	stp	q1, q2, [x1, #64]
	ldp	q1, q2, [x0, #96]
	fmaxnm.4s	v1, v1, v0
	fmaxnm.4s	v0, v2, v0
	stp	q1, q0, [x1, #96]
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
