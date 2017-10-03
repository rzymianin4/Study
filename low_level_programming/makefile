all: test.x

mlib.so: mlib.asm
	nasm -g -felf64  mlib.asm -o mlib.o
	ld -shared mlib.o -o mlib.so

test.x: test.c mlib.so mlib.h
	gcc -g test.c mlib.so -o test.x

test: test.x
	export LD_LIBRARY_PATH=. && ./test.x

clean:
	rm -f *.o *.so test*.x

