CXX=g++
CPPFLAGS=-Wall -Wpedantic -Wextra -std=c++14 -g
LDLIBS=-lSDL2

main:		main.cpp window.o reader.o cpu_compute.o
window.o:	window.h window.cpp
reader.o:	reader.h reader.cpp
cpu_compute.o:	cpu_compute.h cpu_compute.cpp

.PHONY: clean

clean:
	-rm main window.o reader.o cpu_compute.o
