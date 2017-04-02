#include "window.h"
#include "reader.h"

#include <iostream>
#include <unistd.h>

typedef enum computation_ver {
	GPU,
	CPU
} computation_ver;

void get_options(int argc, 
		char **argv, 
		const char *& file_name, 
		unsigned int& width, 
		unsigned int& height,
		computation_ver& version,
		bool& show_output);
std::vector<uint32_t> solve(const computation_ver& version,
		std::vector<point_charge_t>& charges,
		unsigned int width,
		unsigned int height);
void usage(char *name);

int main(int argc, char **argv)
{
	unsigned int width = 640, height = 480;
	const char *file_name = "\0";
	computation_ver version = GPU;
	bool show_output = true;

	get_options(argc, argv, file_name, width, height, version, show_output);
	std::vector<point_charge_t> charges;
	try {
		input_reader reader(file_name);
		charges = reader.read_contents();
	} catch (std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	std::vector<uint32_t> result = solve(version, charges, width, height);
	if (show_output) {
		window win("Electric field", width, height);
		win.show_window(result);
	}
	return EXIT_SUCCESS;
}

std::vector<uint32_t> solve(const computation_ver& version,
		std::vector<point_charge_t>& charges,
		unsigned int width,
		unsigned int height)
{
	gpu_computation gpu_solver;
	cpu_computation cpu_solver;
	switch (version) {
		case GPU:
			return gpu_solver.visualization(charges, width, height);
			break;
		case CPU:
			return cpu_solver.visualization(charges, width, height);
			break;
		default:
			throw std::invalid_argument("Unexpected computation type");
			break;
	}
}

void get_options(int argc, 
		char **argv, 
		const char *& file_name, 
		unsigned int& width, 
		unsigned int& height,
		computation_ver& version,
		bool& show_output)
{
	int c = 0;
	while ((c = getopt(argc, argv, "w:h:f:gcs")) != -1) {
		switch (c) {
			case 'w': // width
				width = atoi(optarg);
				break;
			case 'h': // height
				height = atoi(optarg);
				break;
			case 'f': // file
				file_name = optarg;
				break;
			case 'g': // gpu
				version = GPU;
				break;
			case 'c': // cpu
				version = CPU;
				break;
			case 's': // silent
				show_output = false;
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
	std::cerr << "Usage: " << name << " -f INPUT_FILE [-w [WINDOW_WIDTH] -h [WINDOW_HEIGHT] -gcs]" << std::endl;
	exit(EXIT_FAILURE);
}
