; Dawid Tracz
 ;I declare that I am the only author of the following code

BITS 64

	EXTERN	_GLOBAL_OFFSET_TABLE_
	EXTERN	malloc
	EXTERN	realloc
	EXTERN	free
	EXTERN	memcpy

	GLOBAL  count:function
	GLOBAL  store:function
	GLOBAL  retrieve:function
	GLOBAL  top_length:function

	;GLOBAL  get_ptr:function
	;GLOBAL  get_front:function
	;GLOBAL  get_aloc_elems:function

;-------------------------------------------------------------------------------------------

SECTION .data
	GLOBAL  sstatus:data 4 ; 4 is numer of bytes (here: int)
	GLOBAL  mode:data 4

	;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	msg:	DB "INIT", 0xa
	msg1:	DB "ms1", 0xa
	msg2:	DB "ms2", 0xa

	sstatus:        DD 0x0
				; 1 - read from empty queue
				; 2 - bad arguments
				; 3 - bad alloc
	mode:           DD 0x0

	_mem_ptr:       DQ 0x1  ; (void*) ptr to the queue (to the begin of the allocated memory)
	_front:         DQ 0x2  ; (void*) ptr to the first element of queue;
	_data_size:     DQ 0x0  ; (long) defines length of data in bytes;
	                        ;        (_front + _data_size) % _mem_size
	                        ;        points to the past-the-last element
	_mem_size:      DQ 0x0  ; (long) defines length of allocated memory
	                        ;        if equals 0 - queue not initialized yet

		; every memory "cell" takes 16 bytes, FIRST 8 - for the ptr to the byte sequecne
		;                                    SECOND 8 - for this sequecne length (long)
		; | ptr . len | ptr . len | ptr . len | ptr . len | ...

;-------------------------------------------------------------------------------------------

SECTION .text
	default rel

	;  allocate first queue memory or do nothing if already initialized
	 ; if failed sets sstatus to -3
	 ; takes no arguments
	 ; retunrs 0 if success, or already initialized or -1 if bad alloc
	_init:
		sub 	rsp, 0x8
		mov 	rcx, [_mem_size]
		jrcxz	.initialize                         ; jump if not initialized
			xor		rax, rax
			add 	rsp, 0x8
			ret		
		.initialize:
		mov 	rdi, 0x100                          ; start size 0x100 == 16 elements
		mov 	[_mem_size], rdi                    ; set first length
		call	malloc wrt ..plt
		mov 	[_mem_ptr], rax
		test	rax, rax
		jz		.bad_alloc
		.mem_OK:
			mov		[_front], rax
			xor		rax, rax
			add 	rsp, 0x8
			ret
		.bad_alloc:
			mov		rdi, 0x3
			mov 	[_mem_size], rax                ; set _mem_size as 0 (failed initalization)
			call	_set_sstatus                    ; set sstatus to rdi and rax to -1
			add 	rsp, 0x8
			ret


	;  sets sstatus to given value
	 ; unsave registers: {rax, rcx, rdx}
	 ; rdi - status to set
	 ; returns -1
	_set_sstatus:
		sub 	rsp, 0x8
		mov 	rcx, _GLOBAL_OFFSET_TABLE_
		mov 	rdx, [rcx + sstatus wrt ..got]
		mov		rax, -0x1
		mov 	[rdx], rdi
		add 	rsp, 0x8
		ret


	;  get the appropriate end of the queue (based on mode) if exist
	 ; unsave registers: {rax, rcx, r8, r9, r10}
	 ; takes no arguments
	 ; returns ptr to the element or 0 if queue is empty
	_get_end:
		sub		rsp, 0x8
		mov		rcx, [_data_size]
		jrcxz .empty_queue                          ; if rcx is 0, just return 0
			mov		rax, [_front]                   ; get _front to rax
			mov		r8, _GLOBAL_OFFSET_TABLE_
			mov		r9, [r8 + mode wrt ..got]       ; get mode to r9
			and		[r9], dword 0x1
			cmp		[r9], dword 0x0
			je		.ret_fornt                      ; if mode==0 return _front (alredy in rax)
				mov		r9, [_mem_size]
				mov		r10, [_mem_ptr]
				lea		rax, [rax+rcx]              ; in rax (_front + _data_size)
				add		r10, r9
				cmp		rax, r10                    ; if _front+_data_size > _mem_size
				jle		.q_consist
					sub 	rax, r9
				.q_consist:
				sub 	rax, 0x10                   ; to points the last inst.of past-the-last
				add		rsp, 0x8
				ret	
			.ret_fornt:
			add		rsp, 0x8
			ret		
		.empty_queue:
		xor		rax, rax                            ; mov rax, 0x0
		add		rsp, 0x8
		ret


	;  if (_data_size == _mem_size) reallocates memory (size *=2), sets sstatus if bad alloc
	 ; if data is 'inconsistent' copy last part to the end of new memory block
	 ; |D|A|B|C| -> |D|A|B|C|.|.|.|.| -> |D|a|b|c|.|A|B|C|
	 ; sets new _front, _mem_size and _mem_ptr (if needed)
	 ; takes no arguments
	 ; returns 0 if OK and -1 if failed to allocate memory
	_expand_mem:
		push	rbx
		push	rbp
		sub		rsp, 0x8

		mov		rcx, [_data_size]
		mov		rbx, [_mem_size]
		xor		rax, rax                            ; mov rax, 0x0
		cmp		rcx, rbx                            ; if _data_size < _mem_size
		jl		.return                           ; no reallocation needed

			mov		rdi, [_mem_ptr]             ; rdi - _mem_ptr
			mov		rbp, [_front]
			lea		rsi, [2*rbx]                ; rsi - new _mem_size
			sub		rbp, rdi                        ; in rbp _front offset (rel. _mem_ptr)
			call	realloc wrt ..plt           ; call realloc(_mem_ptr, new_mem_size)
			test	rax, rax
			jz		.bad_alloc                      ; rax == 0 if realloc failed
				; reallocation success:
				; rax(new ptr);   rbx(old _mem_size)   ; rbp(_front offset)
				lea		r8, [rax+rbp]               ; in r8 new _front (ptr)
				lea		r9, [2*rbx]                 ; in r9 new _mem_size
				mov		[_mem_ptr], rax
				mov		[_mem_size], r9

				test	rbp, rbp                    ; if _front offset > 0
				jz		.front_OK                   ; data since _front needs to be move
					mov		rdx, rbx                ; ()
					lea		rdi, [r8 + rbx]     ; dest ptr
					mov		rsi, r8             ; src ptr
					sub		rdx, rbp            ; num. of bytes to copy (old_mem_size-f_offset)
					mov		[_front], rdi
					call	memcpy wrt ..plt    ; call memcpy(dest_ptr, src_ptr, n_bytes)

					xor		rax, rax                ; mov rax, 0x0
					jmp		.return
				
				.front_OK:
					mov     [_front], r8
					xor		rax, rax                ; mov rax, 0x0
					jmp		.return

			.bad_alloc:
			mov		rdi, 0x3
			call	_set_sstatus                    ; sstatus = 3 ; return -1
		.return:
		add		rsp, 0x8
		pop		rbp
		pop		rbx
		ret


	;  saves given data in given place in queue. Doesn't touch variables!!
	 ; rdi - size of data; rsi - ptr to data; rdx - ptr to place in queue
	 ; returns 0 if OK, or -1 if bad alloc (sets sstatus then)
	_enqueue:
		push	rbx
		push	rdi
		push	rsi

		mov		rbx, rdx                                ; in rbx - place in queue

		call	malloc wrt ..plt
		test	rax, rax
		jne		.mem_OK                                 ; if rax==0 (bad alloc)
			mov		rdi, 0x3
			call	_set_sstatus
			add		rsp, 0x10
			pop		rbx
			ret

		.mem_OK:
		pop		rsi                                 ; ptr to data
		pop		rdx                                 ; data length
		mov		[rbx], rax                              ; save ptr in queue
		mov		rdi, rax                            ; in rax ptr to allocated mem
		mov		[rbx+0x8], rdx                          ; save data length in queue
		call	memcpy wrt ..plt
		xor		rax, rax                                ; success - set rax to 0

		pop		rbx
		ret

	;  loads data from given place in queue to address specyfied by given ptr and free memory
	 ; rdi - ptr to load data to; rsi - place in queue (| ptr . len |)
	 ; return unspecified
	_dequeue:
		push	rbx
		movdqa	xmm0, [rsi]                             ; to read from memory only once
		movq	rbx, xmm0                               ; moves low-order qword (ptr)
		punpckhqdq xmm0, xmm0                           ; duplicate high-order xmm to low-order
		movq	rdx, xmm0                               ; alternative: pextrq rdx, xmm0, 1 (len)
		mov		rsi, rbx
		call	memcpy wrt ..plt
		mov		rdi, rbx
		call	free wrt ..plt
		pop		rbx
		ret


	;---------------------------------------------------------------------------------------


	;  get number of elements in queue
	 ; takes no arguments
	 ; returns _data_size/16
	count:
		sub		rsp, 0x8
		;- - - - - - - - - - - - -set sstatus to 0 - - - - - - - - - - - - - - - - - - - -
		mov 	r10, _GLOBAL_OFFSET_TABLE_
		mov 	r11, [r10 + sstatus wrt ..got]
		mov		[r11], dword 0x0
		;- - - - - - - - - - - -initialize if it's not - - - - - - - - - - - - - - - - - -
		call	_init
		test	rax, rax                                ; if(rax == 0)
		jne		.return                                 ; (return if bad alloc)
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		mov		rax, [_data_size]
		shr		rax, 0x4                            ; div by 16 (1 element takes 16 bytes)
		.return:
		add		rsp, 0x8
		ret


	;  get length of byte sequecne on curr top of queue
	 ; takes no arguments
	 ; returns length of byte sequecne, or 0 if queue empty
	top_length:
		sub		rsp, 0x8
		;- - - - - - - - - - - - -set sstatus to 0 - - - - - - - - - - - - - - - - - - - -
		mov 	r10, _GLOBAL_OFFSET_TABLE_
		mov 	r11, [r10 + sstatus wrt ..got]
		mov		[r11], dword 0x0
		;- - - - - - - - - - - -initialize if it's not - - - - - - - - - - - - - - - - - -
		call	_init
		test	rax, rax                                ; if(rax == 0)
		jne		.return                                 ; (return if bad alloc)
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		call	_get_end
		test	rax, rax
		jne		.q_not_empty                            ; if rax==0 return it (queue empty)
			mov		rdi, 0x1
			call	_set_sstatus
			add 	rsp, 0x8
			ret
		.q_not_empty:
		lea		rcx, [rax+0x8]                      ; rax points to curr end; add 0x8
		mov		rax, [rcx]                          ; and get length from memory
		.return:
		add rsp, 0x8
		ret


	;  store given byte string (of given length) in queue. length cannot be 0
	 ; sets sstatus if nessesary (2 if bad arg, 3 if bad alloc)
	 ; rdi - string length; rsi - ptr to string
	 ; return 
	store:
		push	r14
		push	r15
		push	rbx
		mov		r14, rdi                                ; length of data to store
		mov		r15, rsi                                ; ptr to data to store
		;- - - - - - - - - - - - -set sstatus to 0 - - - - - - - - - - - - - - - - - - - -
		mov 	r10, _GLOBAL_OFFSET_TABLE_
		mov 	r11, [r10 + sstatus wrt ..got]
		mov		[r11], dword 0x0
		;- - - - - - - - - - - -initialize if it's not - - - - - - - - - - - - - - - - - -
		call	_init
		test	rax, rax                                ; if(rax != 0)
		jne		.return                                 ; (return if bad alloc)
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		cmp		r14, 0x0                                    ; if (data_length > 0)
		jle		.bad_args
		test	r15, r15
		jz		.bad_args
		jmp		.args_OK
		.bad_args:
			mov		rdi, 0x2                            ; sstatus = 2 for bad args
			call	_set_sstatus
			jmp		.return
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		.args_OK:
		call	_expand_mem
		test	rax, rax
		jne		.return                                 ; if realloc failed, ret rax (already set)

		mov		r8, _GLOBAL_OFFSET_TABLE_
		mov		rcx, [_mem_ptr]
		mov		r9, [r8 + mode wrt ..got]               ; get mode to r9
		mov		r10, [_mem_size]
		mov		rdx, [_front]
		mov		rbx, [_data_size]
		and		[r9], dword 0x1
		cmp		[r9], dword 0x0
		je		.mode0
		.mode1:
			add		rcx, r10
			add		rdx, rbx                            ; rdx -- ptr to past-the-last
			add		rbx, 0x10                           ; increase _data_size
			
			cmp		rdx, rcx
			jl		.pos1_OK                            ; if past-the-last after allocated
				sub		rdx, r10                        ; sub _mem_size (% _mem_size)
			.pos1_OK:
				                                    ; place in q. already in rdx (past-the-last)
			mov		rsi, r15                        ; ptr to data to store
			mov		rdi, r14                        ; length of data to store
			call	_enqueue                        ; call _enqueue(ptr_to_q, ptr_to_dat, size)
			test	rax, rax                            ; rax == -1 if bad_alloc (then return)
			jne		.return

			mov		[_data_size], rbx                   ; save new _data_size calculated before
			xor		rax, rax                            ; returns 0 if OK
			jmp		.return
		
		.mode0:
			sub		rdx, 0x10                           ; place before front (to store data)
	
			cmp		rdx, rcx                            
			jge		.pos0_OK                            ; if place is before _mem_ptr
				add		rdx, r10                        ; add to place _mem_size (% _mem_size)
			.pos0_OK:
			
			mov		rsi, r15                        ; ptr to data to store
			mov		rdi, r14                        ; length of data to store
			add		rbx, 0x10                           ; increase _data_size
			mov		r15, rdx                            ; move new _front to call-save register
			call	_enqueue                        ; call _enqueue(ptr_to_q, ptr_to_dat, size)
			test	rax, rax                            ; rax == -1 if bad_alloc (then return)
			jne		.return
			
			mov		[_front], r15                       ; new _front
			mov		[_data_size], rbx                   ; new _data_size
			xor		rax, rax                            ; returns 0 if OK
			;jmp	.return

		.return:
		pop		rbx
		pop		r15
		pop		r14
		ret


	;  retrieve byte string from the curr end of the queue, and save under given ptr
	 ; if queue is empty sets sstatus as 1
	 ; rdi - ptr to write data from the queue
	 ; return 0 if success or -1 if queue empty
	retrieve:
		push	rbx
		mov		rbx, rdi

		;- - - - - - - - - - - - -set sstatus to 0 - - - - - - - - - - - - - - - - - - - -
		mov 	r10, _GLOBAL_OFFSET_TABLE_
		mov 	r11, [r10 + sstatus wrt ..got]
		mov		[r11], dword 0x0
		;- - - - - - - - - - - -initialize if it's not - - - - - - - - - - - - - - - - - -
		call	_init
		test	rax, rax                                ; if(rax != 0)
		jne		.return                                 ; (return if bad alloc)
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		mov		r11, [_data_size]
		test	r11, r11
		jz		.q_empty
		test	rbx, rbx
		jz		.bad_args
		jmp		.args_OK
		.q_empty:
			mov		rdi, 0x1
			call	_set_sstatus
			jmp		.return
		.bad_args:
			mov		rdi, 0x2
			call	_set_sstatus
			jmp		.return
		;- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		.args_OK:
		mov		r8, _GLOBAL_OFFSET_TABLE_
		mov		r10, [_mem_size]
		mov		rcx, [_mem_ptr]
		mov		r9, [r8 + mode wrt ..got]               ; get mode to r9
		mov		rsi, [_front]
		sub		r11, 0x10
		add		rcx, r10                                ; rcx points past alloc array
		mov		[_data_size], r11                       ; save decremented _data_size
		and		[r9], dword 0x1
		cmp		[r9], dword 0x0
		je		.mode0
		.mode1:
			add		rsi, r11                            ; rsi -- ptr to the last
			mov		rdi, rbx                            ; load *buf from call-save
			cmp		rsi, rcx
			jl		.pos1_OK                            ; if last elem after allocated
				sub		rsi, r10                        ; sub _mem_size (% _mem_size)
			.pos1_OK:
			call	_dequeue
			xor		rax, rax
			jmp		.return

		.mode0:
			lea		rdx, [rsi+0x10]                     ; new _front
			mov		rdi, rbx                            ; load *buf from call-save
			cmp		rdx, rcx                            
			jl		.pos0_OK	                        ; if new _front out of array
				sub		rdx, r10                        ; new _front to the begin
			.pos0_OK:
			mov		[_front], rdx                       ; save new _front
			call	_dequeue                            ; old _front alredy in rsi
			xor		rax, rax
			;jmp		.return

		.return:
		pop		rbx
		ret


	;---------------------------------------------------------------------------------------

	 ;get_ptr:
	 ;	sub		rsp, 0x8
	 ;	mov		rax, [_mem_ptr]
	 ;	add		rsp, 0x8
	 ;	ret

	 ;get_front:
	 ;	sub		rsp, 0x8
	 ;	mov		rax, [_front]
	 ;	add		rsp, 0x8
	 ;	ret

	 ;get_aloc_elems:
	 ;	sub		rsp, 0x8
	 ;	mov		rax, [_mem_size]
	 ;	shr		rax, 0x4
	 ;	add		rsp, 0x8
	 ;	ret
