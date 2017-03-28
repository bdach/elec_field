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
	double x_scaled = bounds->x_min + blockIdx.x * bounds->x_scale / (double)gridDim.x;
	double y_scaled = bounds->y_min + blockIdx.y * bounds->y_scale / (double)gridDim.y;
	double dx = charge.x - x_scaled;
	double dy = charge.y - y_scaled;
	double r = sqrt(dx * dx + dy * dy);
	double intensity = k * charge.charge / r;
	unsigned long offset = stride * (gridDim.x * blockIdx.y + blockIdx.x);
	result[2 * offset + threadIdx.x] = intensity * dx / r;
	result[2 * offset + blockDim.x + threadIdx.x] = intensity * dy / r;
}

__global__ void reduce(double *g_idata, 
		double *g_odata)
{
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[tid];
}

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		double *result) {
	const unsigned int charges_size = charge_count * sizeof(point_charge_t);
	const unsigned int bounds_size = sizeof(bounds_t);
	const unsigned long result_size = 2 * sizeof(double) * charge_count * bounds->width * bounds->height;
	const unsigned long reduced_size = 2 * sizeof(double) * bounds->width * bounds->height;

	point_charge_t *d_charges;
	checkCudaErrors(cudaMalloc((void **)&d_charges, charges_size));
	checkCudaErrors(cudaMemcpy(d_charges, charges, charges_size, cudaMemcpyHostToDevice));

	bounds_t *d_bounds;
	checkCudaErrors(cudaMalloc((void**)&d_bounds, bounds_size));
	checkCudaErrors(cudaMemcpy(d_bounds, bounds, bounds_size, cudaMemcpyHostToDevice));

	double *reduced_vec = (double *)malloc(reduced_size);

	double *d_result_vec;
	checkCudaErrors(cudaMalloc((void**)&d_result_vec, result_size));
	double *d_reduced_vec;
	checkCudaErrors(cudaMalloc((void**)&d_reduced_vec, reduced_size));

	dim3 grid(bounds->width, bounds->height, 1);
	dim3 threads(charge_count, 1, 1);

	calculate_intensity<<< grid, threads >>>(d_charges, charge_count, d_bounds, d_result_vec);
	getLastCudaError("Intensity calculation failed");

	dim3 reduction_grid(2 * bounds->width * bounds->height, 1, 1);
	reduce<<< reduction_grid, threads, sizeof(double) * charge_count >>>(d_result_vec, d_reduced_vec);
	getLastCudaError("Reduction failed");

	checkCudaErrors(cudaFree(d_result_vec));
	checkCudaErrors(cudaMemcpy(reduced_vec, d_reduced_vec, reduced_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bounds->height * bounds->width; i++) {
		double x = reduced_vec[2 * i];
		double y = reduced_vec[2 * i + 1];
		result[i] = sqrt(x * x + y * y);
	}
	
	checkCudaErrors(cudaFree(d_charges));
	checkCudaErrors(cudaFree(d_bounds));
	checkCudaErrors(cudaFree(d_reduced_vec));
	free(reduced_vec);
}

