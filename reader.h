#ifndef READER_H
#define READER_H

#include <fstream>
#include <vector>

#include "compute.h"

class input_reader {
private:
	std::ifstream input_stream;
public:
	input_reader(std::string file_name);
	std::vector<point_charge_t> read_contents();
};

#endif
