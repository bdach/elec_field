#include "compute.h"
#include "math.h"

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#define MIN_INTENSITY 1e-10f
#define MAX_INTENSITY 1e10f
#define THREAD_COUNT 1024

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		uint32_t *result);

__global__ void calculate_intensity(const point_charge_t* charges,
		const bounds_t* bounds,
		float* result) {
	const float k = 8.99e-9f; // Coulomb's constant
	point_charge_t charge = charges[threadIdx.x];
	float x_scaled = bounds->x_min + blockIdx.x * bounds->x_scale / (double)gridDim.x;
	float y_scaled = bounds->y_min + blockIdx.y * bounds->y_scale / (double)gridDim.y;
	float dx = charge.x - x_scaled;
	float dy = charge.y - y_scaled;
	float r = sqrt(dx * dx + dy * dy);
	float intensity = k * charge.charge / r;
	// check coalesced writes here
	unsigned long offset = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x);
	result[2 * offset + threadIdx.x] = intensity * dx / r;
	result[2 * offset + blockDim.x + threadIdx.x] = intensity * dy / r;
}

// thrust
__global__ void add_intensities(float *g_idata, 
		float *g_odata)
{
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// manually unrolled last warp
	unsigned blocksize = blockDim.x;
	if ((blocksize >= 64) && (tid < 32)) sdata[tid] += sdata[tid + 32];
	if ((blocksize >= 32) && (tid < 16)) sdata[tid] += sdata[tid + 16];
	if ((blocksize >= 16) && (tid <  8)) sdata[tid] += sdata[tid +  8];
	if ((blocksize >=  8) && (tid <  4)) sdata[tid] += sdata[tid +  4];
	if ((blocksize >=  4) && (tid <  2)) sdata[tid] += sdata[tid +  2];
	if ((blocksize >=  2) && (tid <  1)) sdata[tid] += sdata[tid +  1];
	if (tid == 0) g_odata[blockIdx.x] = sdata[tid];
}

__global__ void total_intensity(float *g_idata,
		float *g_odata,
		unsigned int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float x = g_idata[2 * i];
		float y = g_idata[2 * i + 1];
		__syncthreads();
		g_odata[i] = fmax(fmin(sqrt(x * x + y * y), MAX_INTENSITY), MIN_INTENSITY);
	}
}

__global__ void intensity_to_color(float *g_idata,
		uint32_t *g_odata,
		const float min,
		const float max,
		int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float diff = max - min;
	float intensity = g_idata[i];
	intensity = fmax(intensity, MIN_INTENSITY);
	intensity = fmin(intensity, MAX_INTENSITY);
	float log = log10(intensity);
	float scaled = (log - min) / diff;
	float hue = (1 - scaled) * 300;
	float h_prim = hue / 60.0;
	float f_x = 1 - fabs(fmod(h_prim, 2.0f) - 1);
	uint8_t x = (uint8_t)(f_x * 0xFF);
	unsigned int rounded_h = (unsigned int) h_prim + 1;
	// one-liner here
	g_odata[i] = x << ((rounded_h % 3) * 8);
	g_odata[i] |= 0xff << (8 * (2 - ((rounded_h / 2) % 3)));
}

extern "C" void run_kernel(const point_charge_t *charges,
		const int charge_count,
		const bounds_t *bounds,
		uint32_t *result)
{
	const unsigned long pixel_count = bounds->width * bounds->height;
	const unsigned int charges_size = charge_count * sizeof(point_charge_t);
	const unsigned int bounds_size = sizeof(bounds_t);
	const unsigned long result_size = 2 * sizeof(float) * charge_count * pixel_count;
	const unsigned long reduced_size = sizeof(uint32_t) * pixel_count;

	point_charge_t *d_charges;
	checkCudaErrors(cudaMalloc((void **)&d_charges, charges_size));
	checkCudaErrors(cudaMemcpy(d_charges, charges, charges_size, cudaMemcpyHostToDevice));

	bounds_t *d_bounds;
	checkCudaErrors(cudaMalloc((void**)&d_bounds, bounds_size));
	checkCudaErrors(cudaMemcpy(d_bounds, bounds, bounds_size, cudaMemcpyHostToDevice));

	float *d_result_vec;
	checkCudaErrors(cudaMalloc((void**)&d_result_vec, result_size));

	dim3 charge_intensity_grid(bounds->width, bounds->height, 1);
	dim3 threads(charge_count, 1, 1);

	calculate_intensity<<< charge_intensity_grid, threads >>>(d_charges, d_bounds, d_result_vec);
	getLastCudaError("Intensity calculation failed");

	dim3 component_intensity_grid(2 * bounds->width * bounds->height, 1, 1);
	unsigned int smem = sizeof(float) * charge_count;
	add_intensities<<< component_intensity_grid, threads, smem >>>(d_result_vec, d_result_vec);
	getLastCudaError("Intensity reduction failed");

	int block_count = pixel_count / THREAD_COUNT + 1;
	dim3 max_thread_grid(block_count, 1, 1);
	total_intensity<<< max_thread_grid, THREAD_COUNT >>>(d_result_vec, d_result_vec, pixel_count);
	getLastCudaError("Total intensity calculation failed");

	thrust::device_ptr<float> intensities_ptr(d_result_vec);

	float min = *thrust::min_element(intensities_ptr, intensities_ptr + pixel_count);
	float max = *thrust::max_element(intensities_ptr, intensities_ptr + pixel_count);

	min = log10(fmax(min, MIN_INTENSITY));
	max = log10(fmin(max, MAX_INTENSITY));

	intensity_to_color<<< max_thread_grid, THREAD_COUNT >>>(d_result_vec, (uint32_t*)d_result_vec, min, max, pixel_count);
	getLastCudaError("Conversion to color failed");

	checkCudaErrors(cudaMemcpy(result, (uint32_t*)d_result_vec, reduced_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_charges));
	checkCudaErrors(cudaFree(d_bounds));
	checkCudaErrors(cudaFree(d_result_vec));
}

