#include "cpu_compute.h"
#include <algorithm>
#include <cmath>

#define MIN_LOG_VAL -10
#define MAX_VAL 1e16

std::vector<uint32_t> cpu_computation::visualization(
		std::vector<point_charge_t>& charges, 
		unsigned int width,
		unsigned int height,
		Uint32 pixel_format) {
	std::vector<double> intensities(width * height);
	m_min_intensity = m_max_intensity = 0.0;
	set_scale(charges);
	m_width = width;
	m_height = height;
	for (unsigned y = 0; y < height; ++y) {
		for (unsigned x = 0; x < width; ++x) {
			intensities[y * width + x] = calculate_intensity(charges, x, y);
		}
	}
	m_min_intensity = *std::min_element(intensities.begin(), intensities.end());
	m_max_intensity = *std::max_element(intensities.begin(), intensities.end());
	return to_color(intensities, pixel_format);
}

void cpu_computation::set_scale(std::vector<point_charge_t>& charges) {
	auto cmp_x = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.x < c2.x; };
	auto cmp_y = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.y < c2.y; };
	double x_min = std::min_element(charges.begin(), charges.end(), cmp_x)->x;
	double x_max = std::max_element(charges.begin(), charges.end(), cmp_x)->x;
	double y_min = std::min_element(charges.begin(), charges.end(), cmp_y)->y;
	double y_max = std::max_element(charges.begin(), charges.end(), cmp_y)->y;
	m_x_min = x_min - (x_max - x_min) * 0.1;
	m_x_scale = (x_max - x_min) * 1.2;
	m_y_min = y_min - (y_max - y_min) * 0.1;
	m_y_scale = (y_max - y_min) * 1.2;
}

double cpu_computation::calculate_intensity(
		std::vector<point_charge_t>& charges,
		unsigned int x,
		unsigned int y) {
	double total_intensity = 0.0;
	for (auto charge : charges) {
		double x_scaled = m_x_min + x * m_x_scale / (double)m_width;
		double y_scaled = m_y_min + y * m_y_scale / (double)m_height;
		double r_squared = (x_scaled - charge.x) * (x_scaled - charge.x) + (y_scaled - charge.y) * (y_scaled - charge.y);
		double intensity = k * fabs(charge.charge) / r_squared;
		if (intensity < MAX_VAL) total_intensity += intensity;
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
