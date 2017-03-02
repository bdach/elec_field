#include "cpu_compute.h"
#include <cmath>

std::vector<uint32_t> cpu_computation::visualization(
		std::vector<point_charge_t>& charges, 
		unsigned int width,
		unsigned int height,
		Uint32 pixel_format) {
	std::vector<double> intensities(width * height);
	m_min_intensity = m_max_intensity = 0.0;
	m_width = width;
	m_height = height;
	for (unsigned y = 0; y < height; ++y) {
		for (unsigned x = 0; x < width; ++x) {
			intensities[y * width + x] = calculate_intensity(charges, x, y);
		}
	}
	return to_color(intensities, pixel_format);
}

double cpu_computation::calculate_intensity(
		std::vector<point_charge_t>& charges,
		unsigned int x,
		unsigned int y) {
	double total_intensity = 0.0;
	for (auto charge : charges) {
		double c_x = m_width * charge.x;
		double c_y = m_height * charge.y;
		double r_squared = (x - c_x) * (x - c_x) + (y - c_y) * (y - c_y);
		if (r_squared < 1) continue;
		total_intensity += k * charge.charge / r_squared;
	}
	// this is going to be a race in parallel
	// so far this saves us a constant sequentially
	if (total_intensity < m_min_intensity) {
		m_min_intensity = total_intensity;
	}
	if (total_intensity > m_max_intensity) {
		m_max_intensity = total_intensity;
	}
	return total_intensity;
}

std::vector<uint32_t> cpu_computation::to_color(std::vector<double>& intensities, Uint32 pixel_format) {
	SDL_PixelFormat *format = SDL_AllocFormat(pixel_format);
	std::vector<uint32_t> colors(intensities.size());
	double max_abs_intensity = fmax(fabs(m_min_intensity), m_max_intensity);
	for (unsigned i = 0; i < intensities.size(); ++i) {
		double scaled = intensities[i] / max_abs_intensity;
		if (scaled >= 0.0) {
			colors[i] = SDL_MapRGB(format, (uint8_t) (scaled * 0xff), 0, 0);
		} else {
			colors[i] = SDL_MapRGB(format, 0, 0, (uint8_t) (-scaled * 0xff));
		}
	}
	return colors;
}
