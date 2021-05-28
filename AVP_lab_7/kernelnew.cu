#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <mpi.h>
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <stdint.h>
#include <iomanip>
#include <chrono>

#include "helper_image.h"

using namespace std;

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


__global__ void filter_kernel(unsigned char* inputBitmap, unsigned char* outputBitmap, int height, int width, int channels) {
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

unsigned char* filter_CPU(unsigned char* pixelData, int width, int height, int channels) {
	unsigned char* result = new unsigned char[width * channels * height];
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
			result[pos] = (unsigned char)sum;
		}
	}
	return result;
}

unsigned char* filter_GPU(unsigned char* pixelData, int width, int height, int channels)
{
	size_t size = width * channels * height;
	unsigned char* result = new unsigned char[size];
	unsigned char* pixelDataGPU, * resultGPU;
	CHECK_ERROR(cudaMalloc((void**)&pixelDataGPU, size));
	CHECK_ERROR(cudaMalloc((void**)&resultGPU, size));
	CHECK_ERROR(cudaMemcpy(pixelDataGPU, pixelData, size, cudaMemcpyHostToDevice));
	int gridSize_X = (int)ceil((double)width * channels / (double)BLOCK_SIZE_X);
	int gridSize_Y = (int)ceil((double)height / (double)BLOCK_SIZE_Y);
	dim3 dimGrid(gridSize_X, gridSize_Y);
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	filter_kernel << <dimGrid, dimBlock >> > (pixelDataGPU, resultGPU, height, width, channels);
	cudaMemcpy(result, resultGPU, size, cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaFree(pixelDataGPU));
	CHECK_ERROR(cudaFree(resultGPU));
	CHECK_ERROR(cudaDeviceReset());
	return result;
}

bool isEquals(unsigned char* a, unsigned char* b, int width, int height, int channels) {
	for (int i = 0; i < width * channels; i++)
		for (int j = 0; j < height; j++)
			if (a[i + j * width * channels] != b[i + j * width * channels]) {
				return false;
			}
	return true;
}

using namespace std;

int main(int argc, char** argv) {
	int rank, commsize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);

	const char* imagePath[] = {
	"/home/shared/evm/stud/s8500/Image/ppm/avp_logo.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/belka.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/cat.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/fire.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/graffiti.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/nature.ppm",
	"/home/shared/evm/stud/s8500/Image/ppm/nvidia.ppm",
	};
	const char* outputImageCPU[] = {
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/avp_logo_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/belka_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/cat_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/fire_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/graffiti_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/nature_CPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/nvidia_CPU.ppm" };

	const char* outputImageGPU[] = {
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/avp_logo_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/belka_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/cat_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/fire_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/graffiti_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/nature_GPU.ppm",
	"/home/shared/evm/stud/s8500/u850503/Osetnik_Doroh/test/Images/nvidia_GPU.ppm" };

	vector<int> counts, offset;
	switch (commsize)
	{
	case 1:
		counts.push_back(primaryImagePath.size());
		offset.push_back(0);
		break;
	case 2:
		counts.assign({ (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize });
		offset.assign({ 0, counts[0] });
		break;
	case 3:
		counts.assign({ (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize });
		offset.assign({ 0, counts[0], counts[0] + counts[1] });
		break;
	case 4:
		counts.assign({ (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize });
		offset.assign({ 0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2] });
		break;
	case 5:
		counts.assign({ (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize,((int)primaryImagePath.size() / commsize) });
		offset.assign({ 0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2], counts[0] + counts[1] + counts[2] + counts[3] });
		break;
	case 6:
		counts.assign({ (int)primaryImagePath.size() / commsize + 1, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize });
		offset.assign({ 0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2], counts[0] + counts[1] + counts[2] + counts[3], counts[0] + counts[1] + counts[2] + counts[3] + counts[4] });
		break;
	case 7:
		counts.assign({ (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize, (int)primaryImagePath.size() / commsize });
		offset.assign({ 0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2], counts[0] + counts[1] + counts[2] + counts[3], counts[0] + counts[1] + counts[2] + counts[3] + counts[4],  counts[0] + counts[1] + counts[2] + counts[3] + counts[4] + counts[5] });
		break;
	default:
		exit(0);
	}

	start = MPI_Wtime();
	for (int i = offset[rank]; i < offset[rank] + counts[rank]; i++)
	{
		unsigned int width = 0, height = 0, channels;

		unsigned char* srcData = NULL, * GPUData = NULL, * CPUData = NULL;
		__loadPPM(imagePath[rank], &srcData, &width, &height, &channels);

		cout << "CPU:" << endl;
		auto start_cpu = chrono::steady_clock::now();
		CPUData = filter_CPU(srcData, width, height, channels);
		auto end_cpu = chrono::steady_clock::now();
		auto CPU_time = end_cpu - start_cpu;
		float CPU_count = chrono::duration<double, milli>(CPU_time).count();
		cout << "CPU Time:" << CPU_count << endl;

		GPUData = filter_GPU(srcData, width, height, channels);

		__savePPM(outputImageCPU[rank], CPUData, width, height, channels);
		__savePPM(outputImageGPU[rank], GPUData, width, height, channels);

		free(srcData);
		free(GPUData);
		free(CPUData);
	}
	time = MPI_Wtime() - start;

	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&time, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	if (rank == 0)
	{
		fulltime = avgtime;
		avgtime /= commsize;
		printf("Min: %lf Max: %lf Avg: %lf Full: %lf\n", mintime, maxtime, avgtime, fulltime);
	}

	return 0;
}