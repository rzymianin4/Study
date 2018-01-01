
all : find.x

%.x : %.o
	ld $^ -s -o $@

%.o : %.asm
	nasm $< -felf64 -o $@

.PHONY : all
