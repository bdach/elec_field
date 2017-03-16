CXX=nvcc
CPPFLAGS=-arch compute_50 -Xcompiler "-Wall -Wpedantic -Wextra -std=c++14 -g"
LDLIBS=-lSDL2

main:		main.cpp window.o reader.o cpu_compute.o gpu_compute.o
window.o:	window.h window.cpp
reader.o:	reader.h reader.cpp structs.h
cpu_compute.o:	cpu_compute.h cpu_compute.cpp
gpu_compute.o:	gpu_compute.h gpu_compute.cpp gpu_compute.cu

.PHONY: clean

clean:
	-rm main window.o reader.o cpu_compute.o gpu_compute.o
