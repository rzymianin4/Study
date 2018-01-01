// cyclic queue of string bytes

/*
 * status of last operation:
 * 1 - if that was reading from empty queue
 * 2 - if bad parameters were passed
 * 3 - if memory allocation failed
**/
extern int sstatus;  // status ostatniej operacji

extern int mode;         // current end of queue (mode & 1)


long store(long length, char* buf);		// sore given string (alloc mem, and copy)

long top_length();						// length of string at the current end

long retrieve(char* buf);				// retrieve sting on cuur end (copy to buf)

long count();							// number of strings in the queue

//void* get_ptr();

//void* get_front();

//long get_aloc_elems();
