#include <cstdint>
#include "reader.h"

#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL.h>

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		uint32_t *result);

class gpu_computation {
public:
	std::vector<uint32_t> visualization(
			std::vector<point_charge_t>& charges, 
			unsigned int width, 
			unsigned int height,
			Uint32 pixel_format);
private:
	double m_min_intensity, m_max_intensity;

	double calculate_intensity(
			std::vector<point_charge_t>& charges,
			unsigned int x,
			unsigned int y
			);
	bounds_t set_scale(std::vector<point_charge_t>& charges);
};
