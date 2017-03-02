#include "window.h"
#include "reader.h"

#include <iostream>
#include <unistd.h>

void get_options(int argc, char **argv, const char *& file_name, unsigned int& width, unsigned int& height);
void usage(char *name);

int main(int argc, char **argv) {
	unsigned int width = 640, height = 480;
	const char *file_name = "\0";
	get_options(argc, argv, file_name, width, height);

	input_reader reader(file_name);
	reader.read_contents();

	window win("Electric field", width, height);
	win.show_window();
	return EXIT_SUCCESS;
}

void get_options(int argc, char **argv, const char *& file_name, unsigned int& width, unsigned int& height) {
	int c = 0;
	while ((c = getopt(argc, argv, "w:h:f:")) != -1) {
		switch (c) {
			case 'w':
				width = atoi(optarg);
				break;
			case 'h':
				height = atoi(optarg);
				break;
			case 'f':
				file_name = optarg;
				break;
			default:
				usage(argv[0]);
				break;
		}
	}
	if (strlen(file_name) == 0)
		usage(argv[0]);
}

void usage(char *name) {
	std::cerr << "Usage: " << name << " -f INPUT_FILE [-w [WINDOW_WIDTH] -h [WINDOW_HEIGHT]]" << std::endl;
	exit(EXIT_FAILURE);
}
