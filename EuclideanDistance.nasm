%include "sseutils.nasm"

section .data


q           equ 8
p           equ 12
d           equ 16

k           equ 4

section .bss            ; Sezione contenente dati non inizializzati

section .text			; Sezione contenente il codice macchina

global  EuclideanDistance

EuclideanDistance:

        push	ebp							; salva il Base Pointer
        mov		ebp, esp					; il Base Pointer punta al Record di Attivazione corrente
        push	ebx							; salva i registri da preservare
        push	esi
        push	edi

        mov     eax, 0      ; i = 0 
        mov     edx, [ebp+k]; dim

        xorps xmm6, xmm6
     
fori:
        movaps xmm6, [q + eax]
        subps  xmm6, [p+eax]
        mulps xmm6, xmm6
        haddps xmm6, xmm6
        haddps xmm6, xmm6
        sqrtss xmm6, xmm6
        movss  [d], xmm6
        add     eax, 1  
        cmp     eax, edx
        jl      fori

        pop	edi									; ripristina i registri da preservare
	pop	esi
	pop	ebx
	mov	esp, ebp							; ripristina lo Stack Pointer
	pop	ebp									; ripristina il Base Pointer
	ret										; torna alla funzione C chiamante
