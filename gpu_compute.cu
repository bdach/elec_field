#include "cuda_runtime.h"
#include "structs.h"
#include "math.h"

__global__ void calculate_intensity(const point_charge_t* charges,
		const bounds_t* bounds,
		const intensity_t* intensity) {
	point_charge_t charge = charges[threadIdx.x];
	double x_scaled = bounds->x_min + blockIdx.x * bounds->x_scale / bounds->x_width;
	double y_scaled = bounds->y_min + blockIdx.y * bounds->y_scale / bounds->y_width;
	double dx = charge.x - x_scaled;
	double dy = charge.y - y_scaled;
	double r = sqrt(dx * dx + dy * dy);
	
}
