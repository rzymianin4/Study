; Dawid Tracz
 ; oświadczam, iż poniższy kod został napisany w pełni samodzielnie

BITS 64

segment .data
      hex:    DB "0123456789abcdef", 0xa

	find:	DB "find: `", 0x0
	nsfod:	DB "': No such file or directory", 0xa
	nad:	DB "': Not a directory", 0xa
	nl: 	DB "./", 0xa

	curr_path:	resb 0x8050			; name of current directory
	base_path:	resb 0x8050 		; base path to printing
	cont:		resb 0x800050 		; dirend structures (directory contents)

GLOBAL _start

;----------------------------------------

SECTION .text

_start:
	mov 	rax, [rsp]
	cmp		rax, 0x1
	jle  	compl_bad			; check argc

	lea 	rbx, [rsp+0x8]
	mov 	rbp, [rbx+0x8]		; rbp = argv[1]

	mov 	rdi, rbp			
	mov 	rax, 80
	syscall						; go to argv[1] dir

	cmp 	rax, 0x0 			; check ret of cd
	jne 	bad_arg

	mov 	rdi, curr_path
	mov 	rsi, 0x8050
	mov 	rax, 79
	syscall						; get current directory path

 ; - - - - - - - - - - - -

	mov 	rax, 0x0
	mov 	rcx, 0x0
	.loop:						; copy address to base_path
		mov 	cl, [rbp+rax]
		mov 	[base_path+rax], cl
		add 	rax, 0x1 		; ciekawostka3: swap tej i kolejnej psuje kod...
		cmp 	cl, 0x0
		jne 	.loop

	mov 	rdi, 0x0
	mov 	rsi, base_path
	call 	print_				; print current path
	call	endl

	mov 	rdi, 0x0
	mov 	rsi, base_path
	call	len
	mov 	cl,  [base_path+rax-0x1]
	cmp 	cl, '/'
	jne 	.path_OK
		mov 	rdi, base_path
		call 	pop_from_path
	.path_OK:

 ; - - - - - - - - - - - - 

	mov 	rdi, curr_path
	mov 	rsi, cont
	call 	read_dir

	mov 	rcx, [cont+0x30]
	cmp 	rcx, 0x0
	je		exit

	mov 	r15, base_path
	call main

;---J-U-M-P-S------------------------------------------------

exit:
	mov 	rax, 0x3c
	mov 	rdi, 0x0
	syscall

bad_arg:
	cmp 	rax, 0xffffffffffffffec ; is there such file
	jne 	compl_bad
		mov 	rdi, 0x0			; yes, there is
		mov 	rsi, rbp
		call 	len
		lea  	rcx, [rax-0x1]
		mov 	rdx, 0x0
		mov 	dl,  [rbp+rcx]
		cmp 	dl, '/'
		jne 	.noslash	
			mov 	rdi, 0x0		; not a directory
			mov 	rsi, find
			call	print_
			mov 	rdi, 0x0
			mov 	rsi, rbp
			call	print_
			mov 	rdi, 0xa
			mov 	rsi, nad
			call	print_
			call 	endl
			jmp 	exit
		.noslash:					; slash at the end (print argv[1])
		mov 	rdi, 0x0
		mov 	rsi, rbp
		call	print_
		call	endl
		jmp 	exit
	compl_bad:						; no sush file or directory
	mov 	rdi, 0x0
	mov 	rsi, find
	call	print_
	cmp		rbp, 0x0
	je  	.eoi
		mov 	rdi, 0x0 			; if argv[1] != nullptr
		mov 	rsi, rbp
		call	print_
		.eoi:
	mov 	rdi, 0xa
	mov 	rsi, nsfod
	call	print_
	call 	endl
	jmp 	exit

;---F-U-N-C-T-I-O-N-S----------------------------------------

main:
	push	rbx

	;mov 	rdi, 0x0
	;mov 	rsi, base_path
	;call 	print_			; print current path
	;call	endl

	mov 	rdi, curr_path
	mov 	rsi, cont
	call 	read_dir

 ;mov 	r10, base_path
 ;call 	p_reg
 ;mov 	rdi, 0x0
 ;mov 	rsi, base_path
 ;call	print_
 ;call 	endl
 ;jmp 	exit

	mov 	rcx, [cont+0x30]
	cmp 	rcx, 0x0
	je 		.eoc				; if directory empty -- return

	; _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

	lea 	rbx, [cont]
	.rec_ent:
		mov 	rcx, [rbx+0x12]
		cmp  	rcx, 0x0		; no more files
			je  	.eoc
		mov 	rax, [rbx+0xa]
		shr 	rax, 0x30
		mov 	rcx, [rbx+rax-0x8]
		shr 	rcx, 0x38
		cmp 	rcx, 0x4		; filetype == directory
			je 	.is_a_dir
			mov 	rdi, 0x0
			mov 	rsi, base_path
			call 	print_			; print current path
			call 	pslash
			mov 	rdi, 0x0
			lea 	rsi, [rbx+0x12]
			call 	print_ 			; print file
			call	endl	
			jmp 	.next_dir
			.is_a_dir:
		mov 	rcx, [rbx+0x12]
		cmp 	ecx, ".."		; filename == ..
			je  	.next_dir
		cmp 	ecx, 0x002e		; filename == .
			je  	.next_dir

		lea 	rdi, [rbx+0x12]
		mov 	rsi, curr_path
		call 	app_to_path 	; append directory to path
		lea 	rdi, [rbx+0x12]
		mov 	rsi, base_path
		call 	app_to_path 	; append directory to base_path


		mov 	rdi, 0x0
		mov 	rsi, base_path
		call 	print_			; print current path
		call	endl
		call	main			; call main for new path

		mov 	rdi, curr_path
		call 	pop_from_path	; remove directory from path
		mov 	rdi, base_path  
		call 	pop_from_path	; remove directory from base_path

		mov 	rdi, curr_path
		mov 	rsi, cont
		call 	read_dir		; rereade directory

		.next_dir:
		mov 	rax, [rbx+0xa]
		shr 	rax, 0x30
		add 	rbx, rax
		jmp 	.rec_ent
		.eoc:

	pop 	rbx
	ret

app_to_path: ; char* rdi (dirname), char* rsi (current path)
	push 	rbx
	mov 	rbx, rdi

	mov 	rdi, 0x0
	; current path alredy in rsi
	call	len

	lea 	rsi, [rsi+rax]
	mov 	rcx, '/'
	mov 	[rsi], rcx ; append shash
	add 	rsi, 0x1

	.loop:
		mov 	dil, [rbx] 		; &dirname in rbx
		mov 	[rsi], dil
		lea 	rbx, [rbx+0x1]
		add 	rsi, 0x1
		cmp 	dil, 0x0
		jne 	.loop

	pop 	rbx
	ret

pop_from_path: ; char* rdi (current path)
	push 	rbx
	mov 	rbx, rdi

  lea rsi, [rdi] 			; ciekawostka2: ta linijka nic nie robi poza tym, że sprawia, że kod działa poprawnie...

	mov 	rdi, 0x0
	; current path alredy in rsi
	call	len
	lea 	rbx, [rbx+rax-0x1]	; set ptr at last char

	.loop:
		mov 	rax, 0x0
		mov 	al, [rbx]
		mov 	[rbx], dil
		sub 	rbx, 0x1		; ciekawostka: po swapie tej lini i następnej program się krzaczy...
		cmp 	al, 0x2f
		jne 	.loop

	pop 	rbx
	ret

read_dir: ; char* rdi (dirname), char* rsi (buff to write) ; ret rax (=0 if OK)
	push	rbx
	mov 	rbx, rsi
	call 	flush 				; clear buff to write

	; dirname alredy in rdi
	mov 	rdx, 0400o
	mov 	rax, 2
	mov 	rsi, 0o
	syscall						; open directory	

	mov 	rdi, rax
	mov 	rsi, rbx			; (buff to write)
	mov 	rbx, rax			; dirdes in rbx
	mov 	rdx, 0x800050
	mov 	rax, 78
	syscall						; get dirend structure

	mov 	rdi, rbx
	mov 	rbx, rax
	mov 	rax, 3
	syscall						; close directory
	mov 	rax, rbx

	pop 	rbx
	ret

flush: ; char* rsi (buff to clear); ret rax (number of cleared bytes (round to 64bits))
	sub 	rsp, 0x8

	mov 	rax, 0x0
	mov 	rcx, 0x0
	.loop:
		mov 	rdx, [rsi+rax]
		mov 	[rsi+rax], rcx
		add 	rax, 0x8
		cmp 	rdx, 0x0
		jne 	.loop
	sub 	rax, 0x8

	add 	rsp, 0x8
	ret

print_:	; char rdi, char* rsi ; ret int rax (number of printed bytes) ; printig one more!!!
	sub 	rsp, 0x8
	mov 	rax, 0x0
	mov 	rdx, 0x0			; length
	.inc:						; while(rsi[rdx]!=rdi) rdx++;
		mov 	al, [rsi+rdx]
		add 	rdx, 0x1
		cmp 	rdi, rax
		jne 	.inc
	sub 	rdx, 0x1
	mov 	rax, 0x1			; set write
	mov 	rdi, 0x1			; to stdout
	syscall						; texptr alredy in rsi
	mov 	rax, rdx
	add 	rsp, 0x8
	ret

len: 	; char rdi, char* rsi
	sub 	rsp, 0x8
	mov 	rdx, 0x0
	mov 	rax, 0x0			; length
	.inc:						; while(rsi[rdx]!=rdi) rdx++;
		mov 	dl, [rsi+rax]
		add 	rax, 0x1
		cmp 	rdi, rdx
		jne 	.inc
	sub 	rax, 0x1
	add 	rsp, 0x8
	ret

endl: 	; print '\n'
	sub 	rsp, 0x8
	mov 	rax, 0x1 			; set write
	mov 	rdi, 0x1 			; to stdout	
	lea 	rsi, [nl+0x2] 		; textptr
	mov 	rdx, 0x1 			; length
	syscall
	add 	rsp, 0x8
	ret

pslash: ; print "/"
	sub 	rsp, 0x8
	mov 	rax, 0x1 			; set write
	mov 	rdi, 0x1 			; to stdout	
	lea 	rsi, [nl+0x1] 			; textptr
	mov 	rdx, 0x1 			; length
	syscall
	add 	rsp, 0x8
	ret

here: 	; print "./"
	sub 	rsp, 0x8
	mov 	rax, 0x1 			; set write
	mov 	rdi, 0x1 			; to stdout	
	lea 	rsi, [nl] 			; textptr
	mov 	rdx, 0x2 			; length
	syscall
	add 	rsp, 0x8
	ret

;---T-H-R-A-S-H-E-S------------------------------------------

p_reg:
      push    r11
      push    rax
      push    rdi
      push    rsi
      push    rdx
      mov     rax, 0x1                        ;set write
      mov     rdi, 0x1                        ;to stdout
      mov     rdx, 0x1                        ;how long
      mov             r14, 0x10
      .xxx:
      mov             r11, r10
      ;shl            r11, 0x3c
      shr             r11, 0x3c
      lea     rsi, [hex+r11]
      syscall
      shl             r10, 0x4
      sub             r14, 0x1
      cmp             r14, 0x0
      jg .xxx
      lea     rsi, [hex+0x10]         ;what
      syscall
      pop             rdx
      pop             rsi
      pop             rdi
      pop             rax
      pop             r11
      ret
