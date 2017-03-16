#include <cstdint>
#include "reader.h"

#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL.h>

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		double *result);

class gpu_computation {
public:
	std::vector<uint32_t> visualization(
			std::vector<point_charge_t>& charges, 
			unsigned int width, 
			unsigned int height,
			Uint32 pixel_format);
private:
	const double k = 8.99e-9; // Coulomb's constant
	double m_min_intensity, m_max_intensity;

	double calculate_intensity(
			std::vector<point_charge_t>& charges,
			unsigned int x,
			unsigned int y
			);
	bounds_t set_scale(std::vector<point_charge_t>& charges);
	std::vector<uint32_t> to_color(std::vector<double>& intensities, Uint32 pixel_format);
	uint32_t hue_to_rgb(double hue, SDL_PixelFormat* format);
};
