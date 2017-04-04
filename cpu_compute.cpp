#include "cpu_compute.h"
#include <algorithm>
#include <cmath>

#define MIN_LOG_VAL -10
#define MAX_LOG_VAL 10

std::vector<uint32_t> cpu_computation::visualization(
		std::vector<point_charge_t>& charges, 
		unsigned int width,
		unsigned int height) {
	std::vector<float> intensities(width * height);
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
	return to_color(intensities);
}

void cpu_computation::set_scale(std::vector<point_charge_t>& charges) {
	auto cmp_x = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.x < c2.x; };
	auto cmp_y = [](const point_charge_t& c1, const point_charge_t& c2) { return c1.y < c2.y; };
	float x_min = std::min_element(charges.begin(), charges.end(), cmp_x)->x;
	float x_max = std::max_element(charges.begin(), charges.end(), cmp_x)->x;
	float y_min = std::min_element(charges.begin(), charges.end(), cmp_y)->y;
	float y_max = std::max_element(charges.begin(), charges.end(), cmp_y)->y;
	m_x_min = x_min - (x_max - x_min) * 0.1;
	m_x_scale = (x_max - x_min) * 1.2;
	m_y_min = y_min - (y_max - y_min) * 0.1;
	m_y_scale = (y_max - y_min) * 1.2;
}

float cpu_computation::calculate_intensity(
		std::vector<point_charge_t>& charges,
		unsigned int x,
		unsigned int y) {
	float intensity_x = 0.0, intensity_y = 0.0;
	for (auto charge : charges) {
		float x_scaled = m_x_min + x * m_x_scale / (float)m_width;
		float y_scaled = m_y_min + y * m_y_scale / (float)m_height;
		float dx = charge.x - x_scaled;
		float dy = charge.y - y_scaled;
		float r = sqrt(dx * dx + dy * dy);
		float intensity = k * charge.charge / r;
		intensity_x += intensity * dx / r;
		intensity_y += intensity * dy / r;
	}
	auto result = fmax(sqrt(intensity_x * intensity_x + intensity_y * intensity_y), 1e-10);
	return result;
}

std::vector<uint32_t> cpu_computation::to_color(std::vector<float>& intensities) {
	std::vector<uint32_t> colors(intensities.size());
	m_min_intensity = fmax(log10(m_min_intensity), MIN_LOG_VAL);
	m_max_intensity = fmin(log10(m_max_intensity), MAX_LOG_VAL);
	float diff = m_max_intensity - m_min_intensity;
	for (unsigned i = 0; i < intensities.size(); ++i) {
		float log = fmax(log10(intensities[i]), MIN_LOG_VAL);
		float scaled = (log - m_min_intensity) / diff;
		float hue = (1 - scaled) * 300;
		colors[i] = hue_to_rgb(hue);
	}
	return colors;
}

uint32_t cpu_computation::hue_to_rgb(float hue) {
	float h_prim = hue / 60.0f;
	float f_x = 1 - fabs(fmod(h_prim, 2.0f) - 1);
	uint8_t x = (uint8_t)(f_x * 0xFF);
	unsigned int rounded_h = (unsigned int) h_prim + 1;
	uint32_t color = 0;
	color |= x << ((rounded_h % 3) * 8);
	color |= 0xff << (8 * (2 - ((rounded_h / 2) % 3)));
	return color;
}
