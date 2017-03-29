#include "gpu_compute.h"
#include <algorithm>
#include <cmath>

#define MIN_LOG_VAL -10
#define MAX_VAL 1e10

std::vector<uint32_t> gpu_computation::visualization(
		std::vector<point_charge_t>& charges, 
		unsigned int width,
		unsigned int height,
		Uint32 pixel_format) {
	std::vector<uint32_t> intensities(width * height);
	bounds_t bounds = set_scale(charges);
	bounds.width = width;
	bounds.height = height;

	run_kernel(&charges[0], charges.size(), &bounds, &intensities[0]);
	return intensities;
}

bounds_t gpu_computation::set_scale(std::vector<point_charge_t>& charges) {
	bounds_t result;
	auto cmp_x = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.x < c2.x; };
	auto cmp_y = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.y < c2.y; };
	double x_min = std::min_element(charges.begin(), charges.end(), cmp_x)->x;
	double x_max = std::max_element(charges.begin(), charges.end(), cmp_x)->x;
	double y_min = std::min_element(charges.begin(), charges.end(), cmp_y)->y;
	double y_max = std::max_element(charges.begin(), charges.end(), cmp_y)->y;
	result.x_min = x_min - (x_max - x_min) * 0.1;
	result.x_scale = (x_max - x_min) * 1.2;
	result.y_min = y_min - (y_max - y_min) * 0.1;
	result.y_scale = (y_max - y_min) * 1.2;
	return result;
}
