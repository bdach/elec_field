#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "reader.h"

#define BUF_SIZE 64

input_reader::input_reader(std::string file_name) : input_stream(file_name) {
}

std::vector<point_charge_t> input_reader::read_contents() {
	std::vector<point_charge_t> charges;
	char buf[BUF_SIZE];
	char *token;
	while (!input_stream.eof()) {
		point_charge_t p;
		input_stream.getline(buf, BUF_SIZE);
		if (buf[0] == '#') continue;
		token = strtok(buf, ",");
		if (token == nullptr) break;
		p.x = atof(token);
		token = strtok(nullptr, ",");
		if (token == nullptr) break;
		p.y = atof(token);
		token = strtok(nullptr, ",");
		if (token == nullptr) break;
		p.charge = atof(token);
		std::cerr << p.x << "," << p.y << "," << p.charge << std::endl;
		charges.push_back(p);
	}
	return charges;
}
