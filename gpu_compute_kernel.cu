#include "structs.h"
#include "math.h"

#include "cuda_runtime.h"
#include "helper_cuda.h"

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		double *result);

__global__ void calculate_intensity(const point_charge_t* charges,
		const int stride,
		const bounds_t* bounds,
		double* result) {
	const double k = 8.99e-9; // Coulomb's constant
	point_charge_t charge = charges[threadIdx.x];
	double x_scaled = bounds->x_min + blockIdx.x * bounds->x_scale / (double)bounds->width;
	double y_scaled = bounds->y_min + blockIdx.y * bounds->y_scale / (double)bounds->height;
	double dx = charge.x - x_scaled;
	double dy = charge.y - y_scaled;
	double r = sqrt(dx * dx + dy * dy);
	double intensity = k * charge.charge / r;
	unsigned long offset = stride * (bounds->width * blockIdx.y + blockIdx.x);
	result[2 * (threadIdx.x + offset)] = intensity * dx / r;
	result[2 * (threadIdx.x + offset) + 1] = intensity * dy / r;
}

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		double *result) {
	const unsigned int charges_size = charge_count * sizeof(point_charge_t);
	const unsigned int bounds_size = sizeof(bounds_t);
	const unsigned long result_size = 2 * sizeof(double) * charge_count * bounds->width * bounds->height;

	point_charge_t *d_charges;
	checkCudaErrors(cudaMalloc((void **)&d_charges, charges_size));
	checkCudaErrors(cudaMemcpy(d_charges, charges, charges_size, cudaMemcpyHostToDevice));

	bounds_t *d_bounds;
	checkCudaErrors(cudaMalloc((void**)&d_bounds, bounds_size));
	checkCudaErrors(cudaMemcpy(d_bounds, bounds, bounds_size, cudaMemcpyHostToDevice));

	double *result_vec = (double *)malloc(result_size);

	double *d_result_vec;
	checkCudaErrors(cudaMalloc((void**)&d_result_vec, result_size));

	dim3 grid(bounds->width, bounds->height, 1);
	dim3 threads(charge_count, 1, 1);
	calculate_intensity<<< grid, threads >>>(d_charges, charge_count, d_bounds, d_result_vec);

	getLastCudaError("Kernel failed");

	checkCudaErrors(cudaMemcpy(result_vec, d_result_vec, result_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bounds->height * bounds->width; i++) {
		double x = 0, y = 0;
		for (int j = 0; j < charge_count; j++) {
			x += result_vec[2 * (i * charge_count + j)];
			y += result_vec[2 * (i * charge_count + j) + 1];
		}
		result[i] = sqrt(x * x + y * y);
	}
	
	checkCudaErrors(cudaFree(d_charges));
	checkCudaErrors(cudaFree(d_bounds));
	checkCudaErrors(cudaFree(d_result_vec));
	free(result_vec);
}

