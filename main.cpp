#include "window.h"

#include <iostream>
#include <unistd.h>

void get_options(int argc, char **argv, unsigned int& width, unsigned int& height);
void usage(char *name);

int main(int argc, char **argv) {
	unsigned int width = 640, height = 480;
	get_options(argc, argv, width, height);
	window win("Electric field", width, height);
	win.show_window();
	return EXIT_SUCCESS;
}

void get_options(int argc, char **argv, unsigned int& width, unsigned int& height) {
	int c = 0;
	while ((c = getopt(argc, argv, "w:h:")) != -1) {
		switch (c) {
			case 'w':
				width = atoi(optarg);
				break;
			case 'h':
				height = atoi(optarg);
				break;
			default:
				usage(argv[0]);
				break;
		}
	}
}

void usage(char *name) {
	std::cerr << "Usage: " << name << " -w [WINDOW_WIDTH] -h [WINDOW_HEIGHT" << std::endl;
	exit(EXIT_FAILURE);
}
