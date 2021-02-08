#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <windows.h>
#include <ctime>


#define GREEN 0x02
#define RED 0x0C
#define WHITE 0x07

using namespace std;


void fill(size_t height, size_t width, float **matrix, int key) {
	srand(clock() + key);
	for (int i = 0; i < height; i++)
		for(int j=0; j < width ;j++)
			matrix[i][j] = 1 + rand() % 15;
}

void auto_vec_mul(int height_A, int width_B, int width_A, float** A, float** B, float** C)
{
	DWORD start_time = GetTickCount();
	for (int i = 0; i < height_A; i++)
	{
		float* c = C[i];
		for (int k = 0; k < width_A; k++)
		{
			const float* b = B[k];
			float a = A[i][k];
			for (int j = 0; j < width_B; j++)
				c[j] += a * b[j];
		}
	}
	cout << setw(5) << "Time of auto vectorized multiplication: " << GetTickCount() - start_time << "ms" << endl;
}


void no_vec_mul(int height_A, int width_B, int width_A, float** A, float** B, float** C)
{
	DWORD start_time = GetTickCount();
	for (int i = 0; i < height_A; i++)
	{
		float* c = C[i];
		for (int k = 0; k < width_A; k++)
		{
			const float* b = B[k];
			float a = A[i][k];
#pragma loop(no_vector)
			for (int j = 0; j < width_B; j++)
				c[j] += a * b[j];
		}
	}
	cout << setw(5) << "Time of non vectorized multiplication: " << GetTickCount() - start_time << "ms" << endl;
}


void AVX_mul(int height_A, int width_B, int width_A, float** A, float** B, float** C)
{
	DWORD start_time = GetTickCount();
	for (int i = 0; i < height_A; i++)
	{
		float* c = C[i];
		for (int k = 0; k < width_A; k++)
		{
			const float* b = B[k];
			const float a = A[i][k];
			for (int j = 0; j < width_B; j += 8) {
				_mm256_storeu_ps(c + j, _mm256_add_ps(_mm256_loadu_ps(c + j), _mm256_mul_ps(_mm256_set1_ps(a), _mm256_loadu_ps(b + j))));
			}
		}
	}
	cout << setw(5) << "Time of manually vectorized multiplication: " << GetTickCount() - start_time << "ms" << endl;
}


void print(size_t height,size_t width, float **matrix)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			cout << setw(4) << matrix[i][j] << "\t";
		cout << endl;
	}
}

void changeColors(int color)
{
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, (WORD)((0 << 4) | color));
}

#define SIZE 500

bool check(float** matrix_C, float** matrix_C_AVX)
{
	for (int i = 0; i < SIZE * 4; i++)
		for (int j = 0; j < SIZE * 4; j++)
		{
			if (matrix_C[i][j] == matrix_C_AVX[i][j])
				return TRUE;
			else
				return FALSE;
		}
}

int main()
{
	float** matrix_A, ** matrix_B, ** matrix_C, ** matrix_C_AVX;


	matrix_A = new float* [SIZE * 4];
	for (int i = 0; i < SIZE * 4; i++) {
		matrix_A[i] = new float[SIZE * 8];
		ZeroMemory(matrix_A[i], SIZE * 8 * sizeof(float));
	}


	matrix_B = new float* [SIZE * 8];
	for (int i = 0; i < SIZE * 8; i++) {
		matrix_B[i] = new float[SIZE * 4];
		ZeroMemory(matrix_B[i], SIZE * 4 * sizeof(float));
	}

	matrix_C = new float* [SIZE * 4];
	for (int i = 0; i < SIZE * 4; i++) {
		matrix_C[i] = new float[SIZE * 4];
		ZeroMemory(matrix_C[i], SIZE * 4 * sizeof(float));
	}

	matrix_C_AVX = new float* [SIZE * 4];
	for (int i = 0; i < SIZE * 4; i++) {
		matrix_C_AVX[i] = new float[SIZE * 4];
		ZeroMemory(matrix_C_AVX[i], SIZE * 4 * sizeof(float));
	}

	fill(SIZE * 4, SIZE * 8, matrix_A, 14);

	fill(SIZE * 8, SIZE * 4, matrix_B, 43);

	no_vec_mul(SIZE * 4, SIZE * 4, SIZE * 8, matrix_A, matrix_B, matrix_C);
	for (int i = 0; i < SIZE * 4; i++)
		ZeroMemory(matrix_C[i], SIZE * 4 * sizeof(float));

	auto_vec_mul(SIZE * 4, SIZE * 4, SIZE * 8, matrix_A, matrix_B, matrix_C);

	AVX_mul(SIZE * 4, SIZE * 4, SIZE * 8, matrix_A, matrix_B, matrix_C_AVX);

	if (check(matrix_C, matrix_C_AVX) == 1) {
	changeColors(GREEN);
	cout << endl << "ALL RIGHT" << endl;
	changeColors(WHITE);
	}
	else {
		changeColors(RED);
		cout << "WAS MISTAKE" << endl;
		changeColors(WHITE);
	}
	return 0;
}
