#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <numeric>
#include "compat.h"
#include "thrust/device_vector.h"
#include "parameters.cuh"
#include "curand_kernel.h"
#include <random>
#include <thrust/extrema.h>
#include <thrust/count.h>
#include <functional>
#include <fstream>

__global__ void gpu_random_initialize(long long seed, curandState* states, ulong total_threads) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= total_threads) return;

	curand_init(seed + id, 0, 0, &states[id]);  // 	Initialize CURAND
}

__global__ void initializeCodebook(float* codeBook, curandState* states, ulong matrixSize) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= matrixSize) return;

	curandState localState = states[id];

	for (ulong i = id; i < matrixSize; i += stride)
		codeBook[i] = 0.9998 * curand_uniform(&localState) + 0.0001;
	states[id] = localState;
}

void display_codebook(thrustHvec<float> codebook) {
	std::cout << std::endl << "Codebook weights:\t";
	for(int mesh = 0; mesh < std::min(showMatrixLimit, numberOfMeshCategories); mesh++){
		std::cout << "Mesh " << mesh << ":" << std::endl;
		for(int node = 0; node < std::min(showMatrixLimit, dimX * dimY); node++)
			std::cout << node << "\t" << codebook[node * numberOfMeshCategories + mesh] << "\t";
		std::cout << std::endl;
	}
}

void save_codebook(thrustDvec<float> codebook, std::string outFile = "codebook.txt") {
	std::ofstream fout(outFile);
	fout << std::setprecision(5);

	for (int i = 0; i < dimX * dimY * numberOfMeshCategories; i++) {
		fout << codebook[i] << "\t";
		if ((i + 1) % numberOfMeshCategories == 0) fout << std::endl;
	}

	fout.close();
}

void display_article_bmu_interaction(thrustDvec<float> codebook, thrustDvec<ulong> bmu1, thrustDvec<ulong> bmu2, ulong article_filter) {
	for (ulong article = 0; article < numberOfArticles; article += numberOfArticles/10) {
		std::cout << "Article Number: " << article << "\tBMU1:\t" << bmu1[article] << "\tBMU2:\t" << bmu2[article] << std::endl;
	}
}