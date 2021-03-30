#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <windows.h>
#include <cuda.h>
#include <curand.h>

#define rows 10000
#define columns 51200

__device__ __constant__ int COLUMNS_DEV;
__device__ __constant__ int ROWS_DEV;


using namespace std;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG) || defined(CUDA_DEBUG)
	if (result != cudaSuccess)
		cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
#endif
	return result;
}

inline
curandStatus_t checkCurand(curandStatus_t result)
{
#if defined(DEBUG) || defined(_DEBUG) || defined(CUDA_DEBUG)
	if (result != CURAND_STATUS_SUCCESS)
		cerr << "CURAND Runtime Error" << endl;
#endif
	return result;
}

void cudaRand(char* matrix) {
	curandGenerator_t gen;
	char* devData;

	checkCuda(cudaMalloc(&devData, rows * columns));

	checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurand(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	checkCurand(curandGenerate(gen, (unsigned int*)devData, rows * columns / sizeof(unsigned int)));

	checkCuda(cudaMemcpy(matrix, devData, rows * columns, cudaMemcpyDeviceToHost));
	checkCurand(curandDestroyGenerator(gen));

	checkCuda(cudaFree(devData));

}

void transform(char* source, char* dest)
{
	int offset = 0;
	DWORD64 start = GetTickCount64();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < columns; j += 4)
		{
			dest[i * columns / 4 + j / 4] = source[i * columns + j];
			dest[i * columns / 4 + j / 4 + rows * columns / 4] = source[i * columns + j + 1];
			dest[i * columns / 4 + j / 4 + 2 * rows * columns / 4] = source[i * columns + j + 2];
			dest[i * columns / 4 + j / 4 + 3 * rows * columns / 4] = source[i * columns + j + 3];
		}
	cout << "CPU transform: " << GetTickCount64() - start << " ms" << endl;
}

__global__ void transform_cuda(int* source, int* dest) {
	__shared__ char memory[4][128];
	char buffer[4];

	*(int*)buffer = source[blockIdx.y * COLUMNS_DEV + blockIdx.x * COLUMNS_DEV / gridDim.x + threadIdx.y * blockDim.x + threadIdx.x];

	memory[0][32 * threadIdx.y + threadIdx.x] = buffer[0];
	memory[1][32 * threadIdx.y + threadIdx.x] = buffer[1];
	memory[2][32 * threadIdx.y + threadIdx.x] = buffer[2];
	memory[3][32 * threadIdx.y + threadIdx.x] = buffer[3];

	__syncthreads();

	dest[blockIdx.y * COLUMNS_DEV / 4 + blockIdx.x * COLUMNS_DEV / 4 / gridDim.x + threadIdx.y * COLUMNS_DEV * ROWS_DEV / 4 + threadIdx.x] = *(int*)(memory[threadIdx.y] + 4 * threadIdx.x); //cuda memcheck
}

void transform_cuda(char* source, char* dest) {
	int* dev_source;
	int* dev_result;

	int col = columns / 4;
	int row = rows;

	checkCuda(cudaMalloc(&dev_source, rows * columns));
	checkCuda(cudaMalloc(&dev_result, rows * 4 * columns / 4));

	dim3 dimGrid(columns / (32 * 16), rows);
	dim3 dimBlock(32, 4);

	cudaEvent_t start, stop;
	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));

	checkCuda(cudaEventRecord(start));

	checkCuda(cudaMemcpy(dev_source, source, columns * rows, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(COLUMNS_DEV, &col, sizeof(int)));
	checkCuda(cudaMemcpyToSymbol(ROWS_DEV, &row, sizeof(int)));

	transform_cuda << < dimGrid, dimBlock >> > (dev_source, dev_result);

	checkCuda(cudaMemcpy(dest, dev_result, rows * columns, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stop));
	checkCuda(cudaEventSynchronize(stop));

	float time;
	checkCuda(cudaEventElapsedTime(&time, start, stop));
	cout << "Cuda Transform: " << (int)time << " ms" << endl;

	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

	checkCuda(cudaFree(dev_source));
	checkCuda(cudaFree(dev_result));
}

void print(char* matrix, int square, int width)
{
	for (int i = 1; i <= square; i++)
	{
		cout << setw(5) << (int)matrix[i - 1];
		if (i % width == 0)
			cout << endl;
	}

}

bool checkMatrix(char* matrix1, char* matrix2)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < columns; j++)
			if ((int)matrix1[i * columns + j] != (int)matrix2[i * columns + j])
				return false;
	return true;
}




int main() {

	char* source = new char[rows * columns];
	char* cpu_dest = new char[rows * 4 * columns / 4];
	char* cuda_dest = new char[rows * 4 * columns / 4];
	ZeroMemory(source, rows * columns);
	ZeroMemory(cpu_dest, rows * 4 * columns / 4);
	ZeroMemory(cuda_dest, rows * 4 * columns / 4);

	cudaRand(source);

	transform(source, cpu_dest);
	transform_cuda(source, cuda_dest);
	cout << "SOURCE: " << endl;
	print(source, 256, 16);
	cout << "CPU TRANSFORM matrix: " << endl;
	print(cpu_dest, 256, 16 / 4);
	cout << "CUDA TRANSFORM matrix: " << endl;
	print(cuda_dest, 256, 16 / 4);
	cout << endl;

	if (checkMatrix(cpu_dest, cuda_dest))
		cout << "Matrix are identical" << endl;
	else
		cout << "Cpu matrix differs from cuda matrix" << endl;

	delete[]source;
	delete[]cpu_dest;
	delete[]cuda_dest;
}