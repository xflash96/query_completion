CC=gcc
MKLROOT=/opt/intel/mkl

CFLAGS=-Wall -g -O3 -std=gnu11 -m64 -I${MKLROOT}/include -fopenmp -DTRANSPARENT_TRIE
LDFLAGS=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -Wl,-rpath,${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lrt

ifeq ($(UNAME_S),Linux)
	LDFLAGS += -lrt
endif

all:qcomp
qcomp: qcomp.o model.o

.PHONY:clean
clean: 
	rm model.o qcomp.o qcomp
