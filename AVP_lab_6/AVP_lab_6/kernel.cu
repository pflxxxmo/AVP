#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <stdio.h>
#include <windows.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include "Shlwapi.h"
#include <cuda.h>
#include <iomanip>

#include "helper_image.h"

using namespace std;

#pragma intrinsic(__rdtsc)

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 8

#define CHECK_ERROR( call )             \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result ) {          \
    cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << endl;  \
    exit(1); } \
}

const int kernel[3][3] = {
	{ 1, 1, 1 },
	{ 1, -8, 1 },
	{ 1, 1, 1 },
};
__device__ __constant__ int kernelGPU[3][3] = {
	{ 1, 1, 1 },
	{ 1, -8, 1 },
	{ 1, 1, 1 },
};


__global__ void filter_kernel(BYTE* inputBitmap, BYTE* outputBitmap, int height, int width, int channels) {
	const int xIndex = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
	const int yIndex = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
	if (xIndex >= width * channels || yIndex >= height) 
		return;
	int offsetX, offsetY, absX, absY;
	int sum = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			offsetX = (j * 3 - channels);
			offsetY = (i - 1);
			absX = xIndex + offsetX;
			absY = yIndex + offsetY;
			if (absX < 0 || absX >= width * channels) absX = xIndex;
			if (absY < 0 || absY >= height) absY = yIndex;
			sum += inputBitmap[absX + absY * width * channels] * kernelGPU[i][j];
		}
	}
	if (sum < 0) sum = 0;
	if (sum > 255) sum = 255;
	outputBitmap[xIndex + yIndex * width * channels] = sum;
}

BYTE* filter_CPU(BYTE* pixelData, int width, int height, int channels) {
	BYTE* result = new BYTE[width * channels * height];
	if (result == NULL)
		return NULL;
	int pos;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width * channels; x++)
		{
			int sum = 0;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					int X = x + (j * 3 - channels);
					int Y = y + (i - 1);

					if (X >= width * channels || X < 0) X = x;
					if (Y == height || Y == -1) Y = y;

					pos = width * channels * Y + X;
					int kernelVal = kernel[i][j];
					sum += pixelData[pos] * kernelVal;
				}
			}
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			pos = width * channels * y + x;
			result[pos] = (byte)sum;
		}
	}
	return result;
}

BYTE* filter_GPU(BYTE* pixelData, int width, int height, int channels)
{
	float timeGPU = NULL;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	size_t size = width * channels * height;
	BYTE* result = new BYTE[size];
	BYTE* pixelDataGPU, * resultGPU;
	CHECK_ERROR(cudaMalloc((void**)&pixelDataGPU, size));
	CHECK_ERROR(cudaMalloc((void**)&resultGPU, size));
	CHECK_ERROR(cudaMemcpy(pixelDataGPU, pixelData, size, cudaMemcpyHostToDevice));
	int gridSize_X = (int)ceil((double)width * channels / (double)BLOCK_SIZE_X);
	int gridSize_Y = (int)ceil((double)height / (double)BLOCK_SIZE_Y);
	dim3 dimGrid(gridSize_X, gridSize_Y);
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	CHECK_ERROR(cudaEventRecord(start));
	filter_kernel << <dimGrid, dimBlock >> > (pixelDataGPU, resultGPU, height, width, channels);
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaEventRecord(stop));
	CHECK_ERROR(cudaEventSynchronize(stop));
	CHECK_ERROR(cudaEventElapsedTime(&timeGPU, start, stop));
	cout << setw(30) << left << "GPU time: " << timeGPU << " ms" << endl;
	CHECK_ERROR(cudaEventDestroy(start));
	CHECK_ERROR(cudaEventDestroy(stop));
	CHECK_ERROR(cudaFree(pixelDataGPU));
	CHECK_ERROR(cudaFree(resultGPU));
	CHECK_ERROR(cudaDeviceReset());
	return result;
}

bool isEquals(BYTE* a, BYTE* b, int width, int height, int channels) {
	for (int i = 0; i < width * channels; i++)
		for (int j = 0; j < height; j++)
			if (a[i + j * width * channels] != b[i + j * width * channels]) {
				return false;
			}
	return true;
}

int main() {
	unsigned int width = 0, height = 0, channels;
	const char srcImage[] = "nature.ppm";
	const char imageCPU[] = "imageCPU.pgm";
	const char imageGPU[] = "imageGPU.pgm";
	BYTE* srcData = NULL, * GPUData = NULL, * CPUData = NULL;
	__loadPPM(srcImage, &srcData, &width, &height, &channels);

	auto start_time = __rdtsc();
	CPUData = filter_CPU(srcData, width, height, channels);
	auto end_time = __rdtsc();
	cout << setw(30) << left << "CPU time: " << (end_time - start_time) / 3590000 << " ms" << endl;

	GPUData = filter_GPU(srcData, width, height, channels);

	isEquals(CPUData, GPUData, width, height, channels) ? cout << "equals" << endl : cout << "not equals" << endl;

	__savePPM(imageCPU, CPUData, width, height, channels);
	__savePPM(imageGPU, GPUData, width, height, channels);

	free(srcData);
	free(GPUData);
	free(CPUData);

	return 0;
}