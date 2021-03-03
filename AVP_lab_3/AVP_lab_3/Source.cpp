#include <windows.h>
#include <iostream>
#include <iomanip>
#include <intrin.h>

#pragma intrinsic(__rdtsc)

#define CACHE_SIZE 6 * (1024 * 1024) / sizeof(ull)
#define OFFSET 4 * 1024 * 1024 / sizeof(ull)
#define MAX_ASSOCIATION 20
#define TRIES 100

typedef unsigned long long int ull;

using namespace std;


void init(ull* array, int soc) {

	ZeroMemory(array, MAX_ASSOCIATION * OFFSET);
	if (soc == 1) {
		for (int i = 0; i < CACHE_SIZE-1; i++)
			array[i] = i + 1;
		return;
	}

	int blockSize = CACHE_SIZE % soc == 0 ? CACHE_SIZE / soc : CACHE_SIZE / soc + 1;
	int currentOffset = 0;

	for (int i = 0; i < soc - 1; i++) {

		for (int j = 0; j < blockSize; j++)
			array[currentOffset + j] = currentOffset + OFFSET + j;

		currentOffset += OFFSET;
	}

	for (int i = 0; i < blockSize; i++)
		array[currentOffset + i] = i + 1;

}

ull testRead(ull *array) {

	ull index = 0;
	ull start = __rdtsc();

	for (int i = 0; i < TRIES; i++) {

		do {
			index = array[index];
		} while (index);
	}

	return (__rdtsc() - start) / TRIES;
}

int main() {

	ull* array;
	array = new ull[OFFSET * MAX_ASSOCIATION];
	
	for (int i = 1; i <= MAX_ASSOCIATION; i++) {

		init(array, i);
		cout << setw(2) << i << " : " << setw(8) << testRead(array) << " ticks" << endl;
	}
	return 0;
}