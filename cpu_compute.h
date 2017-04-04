#include <cstdint>
#include <vector>
#include "compute.h"

class cpu_computation {
public:
	std::vector<uint32_t> visualization(
			std::vector<point_charge_t>& charges, 
			unsigned int width, 
			unsigned int height);
private:
	const float k = 8.99e-9f; // Coulomb's constant
	float m_min_intensity, m_max_intensity;
	unsigned int m_width, m_height;
	float m_x_min, m_y_min, m_x_scale, m_y_scale;

	float calculate_intensity(
			std::vector<point_charge_t>& charges,
			unsigned int x,
			unsigned int y);
	void set_scale(std::vector<point_charge_t>& charges);
	std::vector<uint32_t> to_color(std::vector<float>& intensities);
	uint32_t hue_to_rgb(float hue);
};
