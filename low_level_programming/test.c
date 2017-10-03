#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>

#include "mlib.h"

char txt1[] = "e";
char txt2[] = "pq";
char txt3[] = "ijk";
char txt4[] = "wxyz";
char txt5[] = "fghnm";

char* ptr;

/*
void print_queue() {
	void** ptr = get_ptr();
	long len = get_aloc_elems();
	printf("________________________\n");
	printf("front: %p\n", get_front());
	printf("queue len: %ld\n", len);
	printf("- - - - - - - - - - - - \n");
	for(int i=0; i<len; i++) {
		printf("%p  %ld  %ld \n", ptr, (long)*(ptr), (long)*(ptr+0x1));
		ptr += 0x2; }
	printf("________________________\n"); }
//*/
/*------------------------------------------------------------------------------------------*/

int check(long act_val, long corr_val, const char* name) {
	fprintf(stderr, "\t%s: %s", name, act_val==corr_val ? "OK!\n" : "FAILED! ");
	if(act_val != corr_val) {
		fprintf(stderr, "(should be: %ld; is: %ld)\n", corr_val, act_val);
		return -1; }
	return 0; }

int check2(char* act_txt, char* corr_txt, long len, const char* name) {
	int res = 1;
	for(int i=0; i<len; i++) {
		res *= (act_txt[i] == corr_txt[i]); }
	fprintf(stderr, "\t%s: %s", name, res ? "OK!\n" : "FAILED! ");
	if(!res) {
		fprintf(stderr, "(should be: %.*s; is: %.*s)\n", (int)len, corr_txt, (int)len, act_txt); }
	return res - 1; }

int check_q(char* act_txt, char* corr_txt, long len) {
	int res = 1;
	for(int i=0; i<len; i++) {
		res *= (act_txt[i] == corr_txt[i]); }
	return res - 1; }

/*------------------------------------------------------------------------------------------*/

int simpleStore1() {
	printf("simpleStore1:\n");
	int res = 0;
	int s_mode = mode;
	long ret = 0;

	res += check(count(), 0, "count");
	ret |= store(1, txt1);
	ret |= store(2, txt2);
	res += check(count(), 2, "count");

	res += check(top_length(), 2, "top_length in the same mode");
	mode = !mode;
	res += check(top_length(), 1, "top_length in the opposite mode");
	
	res += check(ret, 0, "return values");

	mode = s_mode;
	return res; }


int simpleStore2() {
	printf("simpleStore2:\n");
	int res = 0;
	int s_mode = mode;
	long ret = 0;

	mode = !mode;
	ret |= store(3, txt3);
	ret |= store(4, txt4);
	mode = !mode;
	ret |= store(5, txt5);
	
	res += check(count(), 5, "count");
	res += check(top_length(), 5, "top_length 1");
	mode = !mode;
	res += check(top_length(), 4, "top_length 0");

	res += check(ret, 0, "return values");

	mode = s_mode;
	return res; }


int retrieve1() {
	printf("retrieve1:\n");
	int res = 0;
	int s_mode = mode;
	long ret = 0;

	int n = top_length();
	res += check(n, 5, "top_length");
	ret |= retrieve(ptr);
	res += check2(ptr, txt5, 5, "retrieve5");
	
	n = top_length();
	res += check(n, 2, "top_length");
	ret |= retrieve(ptr);
	res += check2(ptr, txt2, 2, "retrieve2");
	
	mode = !mode;

	n = top_length();
	res += check(n, 4, "top_length");
	ret |= retrieve(ptr);
	res += check2(ptr, txt4, 4, "retrieve4");
	
	n = top_length();
	res += check(n, 3, "top_length");
	ret |= retrieve(ptr);
	res += check2(ptr, txt3, 3, "retrieve3");

	n = top_length();
	res += check(n, 1, "top_length");
	ret |= retrieve(ptr);
	res += check2(ptr, txt1, 1, "retrieve1");

	res += check(count(), 0, "count");

	res += check(ret, 0, "return values");
	
	mode = s_mode;
	return res; }


int sstatus1() {
	printf("sstatus1:\n");
	int res = 0;
	int s_mode = mode;
	mode = 2; // 2 & 1 == 0
	
	res += check(top_length(), -1, "top_length of empty");

	res += check(retrieve(ptr), -1, "retrieve0 return value");
	res += check(sstatus, 1, "retrieve0 sstatus");

	res += check(store(0, ptr), -1, "store0 return value");
	res += check(sstatus, 2, "store0 sstatus");

	res += check(store(0x8000000000000000, ptr), -1, "store overflow return value");
	res += check(sstatus, 2, "store overflow sstatus");

	res += check(store(0x7fffffffffffffff, ptr), -1, "great alloc return value");
	res += check(sstatus, 3, "great alloc sstatus");

	count();
	res += check(sstatus, 0, "sstatus after count()");

	mode = s_mode;
	return res; }


int store3() {
	printf("store3 (40 elems):\n");
	int res = 0;
	int s_mode = mode;
	long ret = 0;

	mode=2;
	for(int i=0; i<9; i++) {
		ret |= store(2, txt2);
		ret |= store(3, txt3); }
	mode++;
	for(int i=0; i<11; i++) {
		ret |= store(4, txt4);
		ret |= store(5, txt5); }

	res += check(count(), 40, "count");
	res += check(ret, 0, "return values");

	mode = s_mode;
	return res; }


int retrieve2() {
	printf("retrieve2 (40 elems):\n");
	int res_ptrs = 0;
	int res_lens = 0;
	int res = 0;
	int s_mode = mode;
	long ret = 0;
	long tl = 0;

	mode++;
	for(int i=0; i<15; i++) {
		tl = top_length();
		ret |= retrieve(ptr);
		if(!(i%2)) {
			res_lens -= (tl != 5);
			res_ptrs += check_q(ptr, txt5, 5); }
		else {
			res_lens -= (tl != 4);
			res_ptrs += check_q(ptr, txt4, 4); } }
	res += check(count(), 25, "count");

	mode++;
	for(int i=0; i<18; i++) {
		tl = top_length();
		ret |= retrieve(ptr);
		if(!(i%2)) {
			res_lens -= (tl != 3);
			res_ptrs += check_q(ptr, txt3, 3); }
		else {
			res_lens -= (tl != 2);
			res_ptrs += check_q(ptr, txt2, 2); } }
	res += check(count(), 7, "count");

	res += check(ret, 0, "return values");
	
	int i = 0;
	for(;;) {
		tl = top_length();
		ret += retrieve(ptr);
		if(sstatus) {
			break; }
		if(!(i%2)) {
			res_lens -= (tl != 4);
			res_ptrs += check_q(ptr, txt4, 4); }
		else {
			res_lens -= (tl != 5);
			res_ptrs += check_q(ptr, txt5, 5); }
		i++; }

	res += check(sstatus, 1, "retrieve0 sstatus");
	res += check(ret, -1, "return values");
	res += check(count(), 0, "count");	
	res += check(tl, -1, "top_length empty");	

	fprintf(stderr, "\t- - - - - - - -\n");
	fprintf(stderr, "\ttotal bad retrieves: %d\n", -res_ptrs);
	fprintf(stderr, "\ttotal bad top_lengths: %d\n", -res_lens);

	mode = s_mode;
	return res + res_lens + res_ptrs; }


int bigStore() {
	printf("store (2^21)k elemts, 1kB each:\n");
	int res = 0;
	int s_mode = mode;
	long ret = 0;

	char mword[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	               "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

	int count = 1<<21;
	fprintf(stderr, "\teach sign equals %d operations", count/128);
	for(int i=0; i<count; i++) {
		if(!(i%(count/128))) {
			fprintf(stderr, "%s+", i%(count/2) ? "" : "\n\t");
			fflush(stderr); }
		ret |= store(1024, mword);
		if(ret) break; }
	for(int i=0; i<count; i++) {
		if(!(i%(count/128))) {
			fprintf(stderr, "%s-", i%(count/2) ? "" : "\n\t");
			fflush(stderr); }
		ret |= retrieve(mword);
		if(ret) break; }
	fprintf(stderr, "\n");

	res += check(ret, 0, "return values");

	mode = s_mode;
	return res; }

/*------------------------------------------------------------------------------------------*/

int main() {
	int errs = 0;
	int curr_errs = 0;
	ptr = malloc(5*sizeof(char));
	printf("start mode: %d\n", mode);
	printf("start sstatus: %d\n\n", sstatus);

	curr_errs = -simpleStore1();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;
	
	curr_errs = -simpleStore2();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;
	
	/*(0-end) { 5 2 1 3 4 } (1-end)*/
	
	curr_errs = -retrieve1();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;
	
	/* queue empty */
	
	curr_errs = -sstatus1();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;

	curr_errs = -store3();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;
	
	/*(0-end) { 5 4 5 4 ..(16 more elems).. 5 4 2 3 2 3 ..(12 more elems).. 2 3 } (1-end)*/	
	
	curr_errs = -retrieve2();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;

	curr_errs = -bigStore();
	printf("Errors: %d\n\n", curr_errs);
	errs += curr_errs;

	//write test with deleted source!!!

	if(errs) {
		printf("\t>>> TOTAL ERRORS DETECTED: %d <<<\n\n", errs); }
	else {
		printf("\t>>> NO ERRORS DETECTED! <<<\n\n"); }

	return 0;
}
