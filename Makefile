build:
	gcc -Wall -Wextra -I/usr/local/include/ backend.c -L/usr/local/lib/ -l:libhackrf.so
