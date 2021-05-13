#include "kernel.cuh"

using namespace std;

int main(int argc, char** argv){
	int rank, commsize;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	
	const char* imagePath[] = {
	"/home/shared/evm/stud/s8500/Image/ppm/avp_logo.ppm  ",
	"/home/shared/evm/stud/s8500/Image/ppm/belka.ppm     ",
	"/home/shared/evm/stud/s8500/Image/ppm/cat.ppm       ",
	"/home/shared/evm/stud/s8500/Image/ppm/fire.ppm      ",
	"/home/shared/evm/stud/s8500/Image/ppm/graffiti.ppm  ",
	"/home/shared/evm/stud/s8500/Image/ppm/nature.ppm    ",
	"/home/shared/evm/stud/s8500/Image/ppm/nvidia.ppm    ",
	};
	const char* outputImageCPU[] = {
	"avp_logo_CPU.ppm",
	"belka_CPU.ppm",
	"cat_CPU.ppm",
	"fire_CPU.ppm",
	"graffiti_CPU.ppm",
	"nature_CPU.ppm",
	"nvidia_CPU.ppm"};
	
	const char* outputImageGPU[] = {
	"avp_logo_GPU.ppm",
	"belka_GPU.ppm",
	"cat_GPU.ppm",
	"fire_GPU.ppm",
	"graffiti_GPU.ppm",
	"nature_GPU.ppm",
	"nvidia_GPU.ppm"};
	
	usigned int width = 0, height = 0, channels;
	
	BYTE* srcData = NULL, * GPUData = NULL, * CPUData = NULL;
	__loadPPM(imagePath[rank], &srcData, &width, &height, &channels);
	
	cout << "CPU:"<< endl;
	auto start_cpu = chrono::steady_clock::now();
	CPUData = filter_CPU(srcData, width, height, channels);
	auto end_cpu = chrono::steady_clock::now();
	auto CPU_time = end_cpu - start_cpu;
	float CPU_count = chrono::duration<double, milli>(cpu_time).count();
	cout << "CPU Time:" << CPU_count << endl;
	
	GPUData = filter_GPU(srcData, width, height, channels);
	
	__savePPM(outputImageCPU[rank], CPUData, width, height, channels);
	__savePPM(outputImageGPU[rank], GPUData, width, height, channels);

	free(srcData);
	free(GPUData);
	free(CPUData);
	
	MPI_Finalize();
	return 0;
}