#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include <windows.h>
#include <ctime>


#define GREEN 0x02
#define RED 0x0C
#define WHITE 0x07

using namespace std;


void fill(size_t height, size_t width, float** matrix, int key) {
	srand(clock() + key);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			matrix[i][j] = 1 + rand() % 15;
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


void changeColors(int color)
{
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, (WORD)((0 << 4) | color));
}

#define SIZE 800
#define BLOCK 64

void cache_AVX_mul(int height_A, int width_B, int width_A, float** A, float** B, float** C)
{
	DWORD start_time = GetTickCount();
	for (int h = 0; h < height_A; h += BLOCK)
		for (int common = 0; common < width_A; common += BLOCK)
			for (int w = 0; w < width_B; w += BLOCK)
				for (int i = 0; i < BLOCK; i++)
				{
					float* c = C[i + h] + w;
					for(int k = 0; k < BLOCK; k++ )
					{
						const float* b = B[k + common] + w;
						float a = A[i + h][k + common];
						for (int j = 0; j < BLOCK; j += 8)
							_mm256_storeu_ps(c + j, _mm256_add_ps(_mm256_loadu_ps(c + j), _mm256_mul_ps(_mm256_set1_ps(a), _mm256_loadu_ps(b + j))));
					}
				}
	cout << setw(5) << "Time of manually vectorized multiplication with using cache: " << GetTickCount() - start_time << "ms" << endl;
}

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

	fill(SIZE * 4, SIZE * 8, matrix_A, 15);

	fill(SIZE * 8, SIZE * 4, matrix_B, 48);

	AVX_mul(SIZE * 4, SIZE * 4, SIZE * 8, matrix_A, matrix_B, matrix_C_AVX);

	cache_AVX_mul(SIZE * 4, SIZE * 4, SIZE * 8, matrix_A, matrix_B, matrix_C);

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

	for (int i = 0; i < SIZE * 4; i++) {
		delete[] matrix_A[i];
		delete[] matrix_C[i];
		delete[] matrix_C_AVX[i];
	}
	delete[] matrix_A;
	delete[] matrix_C;
	delete[] matrix_C_AVX;

	for (int i = 0; i < SIZE * 8; i++)
		delete[] matrix_B[i];
	delete[] matrix_B;

	return 0;
}