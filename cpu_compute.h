#include <cstdint>
#include "reader.h"

#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL.h>

class cpu_computation {
public:
	std::vector<uint32_t> visualization(
			std::vector<point_charge_t>& charges, 
			unsigned int width, 
			unsigned int height,
			Uint32 pixel_format);
private:
	const double k = 8.99e-9; // Coulomb's constant
	double m_min_intensity, m_max_intensity;
	unsigned int m_width, m_height;
	double m_x_min, m_y_min, m_x_scale, m_y_scale;

	double calculate_intensity(
			std::vector<point_charge_t>& charges,
			unsigned int x,
			unsigned int y
			);
	void set_scale(std::vector<point_charge_t>& charges);
	std::vector<uint32_t> to_color(std::vector<double>& intensities, Uint32 pixel_format);
	uint32_t hue_to_rgb(double hue, SDL_PixelFormat* format);
};
