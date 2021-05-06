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

#include "helper_image.h"

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 8

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

__global__ void filter_kernel(BYTE* inputBitmap, BYTE* outputBitmap, int height, int dwordWidth, int width) {
	const int xIndex = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
	const int yIndex = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
	if (xIndex >= dwordWidth || yIndex >= height)
		return;
	int threadAbsX = xIndex * 4;
	uint32_t result = 0;
	for (int k = 0; k < 4; k++)
	{
		int byteAbsX = threadAbsX + k;
		int offsetX, offsetY, absX, absY;
		int sum = 0;

		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				offsetX = (i - 1);
				offsetY = (j - 1);
				absX = byteAbsX + offsetX;
				absY = yIndex + offsetY;
				if (absX < 0 || absX >= width)
					absX = byteAbsX;
				if (absY < 0 || absY >= height)
					absY = yIndex;
				sum += inputBitmap[absX + absY * width] * kernelGPU[j][i];
			}
		}
		if (sum < 0) sum = 0;
		if (sum > 255) sum = 255;
		((byte*)&result)[k] = sum;
	}
	((uint32_t*)outputBitmap)[xIndex + yIndex * dwordWidth] = result;
}

BYTE* filter_CPU(BYTE* pixelData, int width, int height) {
	LARGE_INTEGER start, finish, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	BYTE* result = new BYTE[width * height];
	if (result == NULL)
		return NULL;
	int pos;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int sum = 0;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					int X = x + (j - 1);
					int Y = y + (i - 1);

					if (X == width) X = width - 1;
					if (X == -1) X = 0;
					if (Y == height) Y = height - 1;
					if (Y == -1) Y = 0;

					pos = width * Y + X;
					int kernelVal = kernel[i][j];
					sum += pixelData[pos] * kernelVal;
				}
			}
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			pos = width * y + x;
			result[pos] = (byte)sum;
		}
	}
	QueryPerformanceCounter(&finish);
	double time = (finish.QuadPart - start.QuadPart) / (double)freq.QuadPart;
	printf("\ntime CPU = %lf\n", time);
	return result;
}

BYTE* filter_GPU(BYTE* pixelData, int width, int height)
{
	size_t size = width * height;
	LARGE_INTEGER start, finish, freq;
	BYTE* pixelDataGPU, * resultGPU;
	cudaMalloc((void**)&pixelDataGPU, size);
	cudaMalloc((void**)&resultGPU, size);
	cudaMemcpy(pixelDataGPU, pixelData, size, cudaMemcpyHostToDevice);
	int dwordWidth = (width + 3) / 4;
	int gridSize_X = (int)ceil((double)dwordWidth / (double)BLOCK_SIZE_X);
	int gridSize_Y = (int)ceil((double)height / (double)BLOCK_SIZE_Y);
	dim3 dimGrid(gridSize_X, gridSize_Y);
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	filter_kernel << <dimGrid, dimBlock >> > (pixelDataGPU, resultGPU, height, dwordWidth, width);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&finish);
	double time = (finish.QuadPart - start.QuadPart) / (double)freq.QuadPart;
	printf("\ntime CUDA = %lf\n", time);
	BYTE* result = new BYTE[size];
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);
	cudaFree(pixelDataGPU);
	cudaFree(resultGPU);
	cudaDeviceReset();
	return result;
}

bool isEquals(BYTE* a, BYTE* b, int width, int height) {
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			if (a[i + j * width] != b[i + j * width]) {
				return false;
			}
	return true;
}

int main() {
	unsigned int width = 0, height = 0, channels;
	const char srcImage[] = "image.pgm";
	const char imageCPU[] = "imageCPU2.pgm";
	const char imageGPU[] = "imageGPU2.pgm";
	BYTE* srcData = NULL, * GPUData = NULL, * CPUData = NULL;
	__loadPPM(srcImage, &srcData, &width, &height, &channels);

	CPUData = filter_CPU(srcData, width, height);
	GPUData = filter_GPU(srcData, width, height);

	isEquals(CPUData, GPUData, width, height) ? printf("\nequals\n") : printf("\nnot equals\n");


	__savePPM(imageCPU, CPUData, width, height, channels);
	__savePPM(imageGPU, GPUData, width, height, channels);

	free(srcData);
	free(GPUData);
	free(CPUData);

	system("pause");
	return 0;
}