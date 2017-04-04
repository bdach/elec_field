#include <cstdint>
#include "compute.h"

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		uint32_t *result);

class gpu_computation {
public:
	std::vector<uint32_t> visualization(
			std::vector<point_charge_t>& charges, 
			unsigned int width, 
			unsigned int height);
private:
	bounds_t set_scale(std::vector<point_charge_t>& charges);
};
