#include "cpu_compute.h"
#include <cmath>

#define MIN_LOG_VAL -10

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
		total_intensity += k * fabs(charge.charge) / r_squared;
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
	m_min_intensity = fmax(log10(m_min_intensity), MIN_LOG_VAL);
	m_max_intensity = log10(m_max_intensity);
	double diff = m_max_intensity - m_min_intensity;
	for (unsigned i = 0; i < intensities.size(); ++i) {
		double log = fmax(log10(intensities[i]), MIN_LOG_VAL);
		double scaled = (log - m_min_intensity) / diff;
		double hue = (1 - scaled) * 300;
		colors[i] = hue_to_rgb(hue, format);
	}
	return colors;
}

uint32_t cpu_computation::hue_to_rgb(double hue, SDL_PixelFormat* format) {
	double h_prim = hue / 60;
	double f_x = 1 - fabs(fmod(h_prim, 2) - 1);
	uint8_t x = (uint8_t)(f_x * 0xFF);
	if (h_prim <= 1)
		return SDL_MapRGB(format, 0xFF, x, 0);
	if (h_prim <= 2)
		return SDL_MapRGB(format, x, 0xFF, 0);
	if (h_prim <= 3)
		return SDL_MapRGB(format, 0, 0xFF, x);
	if (h_prim <= 4)
		return SDL_MapRGB(format, 0, x, 0xFF);
	if (h_prim <= 5)
		return SDL_MapRGB(format, x, 0, 0xFF);
	return SDL_MapRGB(format, 0xFF, 0, x);
}
