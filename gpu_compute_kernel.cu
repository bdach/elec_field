#include "structs.h"
#include "math.h"

#include "cuda_runtime.h"
#include "helper_cuda.h"

#define MIN_INTENSITY 1e-10
#define MAX_INTENSITY 1e10

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		double *result);

__global__ void calculate_intensity(const point_charge_t* charges,
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
	unsigned long offset = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x);
	result[2 * offset + threadIdx.x] = intensity * dx / r;
	result[2 * offset + blockDim.x + threadIdx.x] = intensity * dy / r;
}

__global__ void add_intensities(double *g_idata, 
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

__global__ void total_intensity(double *g_idata,
		double *g_odata,
		unsigned int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		double x = g_idata[2 * i];
		double y = g_idata[2 * i + 1];
		g_odata[i] = fmax(fmin(sqrt(x * x + y * y), MAX_INTENSITY), MIN_INTENSITY);
	}
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

	double *d_result_vec;
	checkCudaErrors(cudaMalloc((void**)&d_result_vec, result_size));

	dim3 charge_intensity_grid(bounds->width, bounds->height, 1);
	dim3 threads(charge_count, 1, 1);

	calculate_intensity<<< charge_intensity_grid, threads >>>(d_charges, d_bounds, d_result_vec);
	getLastCudaError("Intensity calculation failed");

	dim3 component_intensity_grid(2 * bounds->width * bounds->height, 1, 1);
	unsigned int smem = sizeof(double) * charge_count;
	add_intensities<<< component_intensity_grid, threads, smem >>>(d_result_vec, d_result_vec);
	getLastCudaError("Intensity reduction failed");

	int block_count = bounds->width * bounds->height / 1024;
	dim3 total_intensity_grid(block_count, 1, 1);
	total_intensity<<< total_intensity_grid, 1024 >>>(d_result_vec, d_result_vec, bounds->width * bounds->height);
	getLastCudaError("Total intensity calculation failed");

	checkCudaErrors(cudaMemcpy(result, d_result_vec, reduced_size / 2, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_charges));
	checkCudaErrors(cudaFree(d_bounds));
	checkCudaErrors(cudaFree(d_result_vec));
}

