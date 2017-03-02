CXX=g++
CPPFLAGS=-Wall -Wpedantic -Wextra -std=c++14 -g
LDLIBS=-lSDL2

main:		main.cpp window.o
window.o:	window.h window.cpp

.PHONY: clean

clean:
	-rm main window.o
