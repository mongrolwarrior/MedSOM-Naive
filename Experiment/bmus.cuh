#include <cuda_runtime_api.h>
#include "curand_kernel.h"
#include "compat.h"
#include <cfloat>
#include <fstream>

//--------------------------------------------------------------------------
// DEVICE FUNCTIONS
//

__global__ void sumWeights(float* sumsOfWeights, float* codebook, ulong numberOfMeshCategories, ulong dimX, ulong dimY) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	ulong numberOfNodes = dimX * dimY;

	if (id >= numberOfNodes) return;

	for (ulong node = id; node < numberOfNodes; node += stride) {
		for (ulong mesh = 0; mesh < numberOfMeshCategories; mesh++)
			sumsOfWeights[node] += codebook[node * numberOfMeshCategories + mesh] * codebook[node * numberOfMeshCategories + mesh];
		for (ulong mesh = 0; mesh < numberOfMeshCategories; mesh++)
			codebook[node * numberOfMeshCategories + mesh] /= sqrt(sumsOfWeights[node]);
	}
}

__global__ void calculate_distances_articles_to_bmus(float* distance1, float* distance2, ulong* bmu1, ulong* bmu2, float* sumsOfWeights, float* codebook, ulong* nnz, ulong* columns, ulong* article_indices, 
		ulong dimX, ulong dimY, ulong numberOfMeshCategories, ulong numberOfArticles) {

	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	ulong numberOfNodes = dimX * dimY;

	for (ulong article = id; article < numberOfArticles; article += stride) {
		float best_distance = FLT_MAX;
		ulong temp_bmu1 = 0;
		float second_best_distance = FLT_MAX;
		ulong temp_bmu2 = 0;
		for (ulong node = 0; node < numberOfNodes; node++){
			float dot_product_article_weights = 0.0;
			for (ulong weight_index = 0; weight_index < nnz[article]; weight_index++)
				// article_indices GIVES INDEX OF FIRST NON-ZERO MESH FOR ARTICLE FROM SPARSE MATRIX
				// columns GIVES ACTUAL MESH INDEX BASED ON FIRST NON-ZERO MESH FOR THAT ARTICLE + THE INDEX IN FOR LOOP
				// codebook GIVES THE ACTUAL WEIGHT BETWEEN THAT MESH AND THE NODE
				dot_product_article_weights += codebook[node * numberOfMeshCategories + columns[article_indices[article] + weight_index]];
			//float distance = sumsOfWeights[node] / weight_scale + nnz[article] - 2 * dot_product_article_weights;
			//float distance = sumsOfWeights[node] / (numberOfMeshCategories / (4 * nnz[article])) + nnz[article] - 2 * dot_product_article_weights;
			//float distance = sumsOfWeights[node] / (numberOfMeshCategories / (2 * nnz[article])) + nnz[article] - 2 * dot_product_article_weights;
			//float distance = sqrt(sumsOfWeights[node]) + nnz[article] - 2 * dot_product_article_weights;
			float distance = nnz[article] - 2 * dot_product_article_weights;
			if(distance < 0.0) return;
			if(distance < best_distance){
				second_best_distance = best_distance;
				temp_bmu2 = temp_bmu1;
				best_distance = distance;
				temp_bmu1 = node;
			}
			else if (distance < second_best_distance) {
				second_best_distance = distance;
				temp_bmu2 = node;
			}
		}
		distance1[article] = best_distance;
		bmu1[article] = temp_bmu1;
		distance2[article] = second_best_distance;
		bmu2[article] = temp_bmu2;
	}
}

__global__ void neighborhoodFnForNode(float* codebook, ulong* bmu1, float* neighborhood_function,
		ulong dimX, ulong dimY, ulong trainingNode, ulong numberOfArticles, ulong radius, float std_coeff = 0.5) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	ulong trainingNodeX = trainingNode % dimX;
	ulong trainingNodeY = trainingNode / dimY;
	float gaussian_std = radius * std_coeff;

	for (ulong article = id; article < numberOfArticles; article += stride) {
		ulong bmu = bmu1[article];
		ulong bmuX = bmu % dimX;
		ulong bmuY = bmu / dimY;
		float x_distance = fmaxf((float)bmuX, (float)trainingNodeX) - fminf((float)bmuX, (float)trainingNodeX);
		float y_distance = fmaxf((float)bmuY, (float) trainingNodeY) - fminf((float)bmuY, (float)trainingNodeY);
		float total_distance = x_distance + y_distance;
		neighborhood_function[article] = exp(-total_distance * total_distance / (2 * gaussian_std * gaussian_std));
	}
}

__global__ void calculateNumeratorsForNode(float* numerators, float* neighborhood_function, ulong* nnz, ulong* indices, ulong* columns,
	ulong numberOfMeshCategories, ulong numberOfArticles) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	for (ulong article = id; article < numberOfArticles; article += stride) {
		for (ulong nnz_index = 0; nnz_index < nnz[article]; nnz_index++)
			atomicAdd(&numerators[columns[indices[article] + nnz_index]], neighborhood_function[article]);
	}
}

/*
__global__ void calculateNumeratorsForNode(float* numerators, float* neighborhood_function, ulong* nnz, ulong* indices, ulong* columns,
	ulong numberOfMeshCategories, ulong numberOfArticles) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	for (ulong article = id; article < numberOfArticles; article += stride) {
		for (ulong nnz_index = 0; nnz_index < nnz[article]; nnz_index++)
			atomicAdd(&numerators[columns[indices[article] + nnz_index]], neighborhood_function[article]);
	}
}
*/

__global__ void calculateNewWeights(float* codebook, float* numerators, 
		ulong dimX, ulong dimY, ulong numberOfMeshCategories, ulong trainingNode, float denominator) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfMeshCategories) return;

	for (unsigned long long mesh = id; mesh < numberOfMeshCategories; mesh += stride) {
		codebook[trainingNode * numberOfMeshCategories + mesh] = ((denominator != 0.0) ? (numerators[mesh] / denominator) : 0.01);
	}
}

__global__ void calculateTopographicError(ulong* bmu1, ulong* bmu2, float* distance_bmu1_2, ulong dimX, ulong dimY, ulong numberOfArticles) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	for (ulong article = id; article < numberOfArticles; article += stride) {
		ulong bmu1X = bmu1[article] % dimX;
		ulong bmu1Y = bmu1[article] / dimY;
		ulong bmu2X = bmu2[article] % dimX;
		ulong bmu2Y = bmu2[article] / dimY;
		float dist_X = fmaxf(bmu1X, bmu2X) - fminf(bmu1X, bmu2X);
		float dist_Y = fmaxf(bmu1Y, bmu2Y) - fminf(bmu1Y, bmu2Y);
		distance_bmu1_2[article] = dist_X + dist_Y;
	}
}

//--------------------------------------------------------------------------
// PREDICATES

struct is_greater_than_1
{
	__host__ __device__
		bool operator()(const float& x)
	{
		return x > 1.0;
	}
};

//--------------------------------------------------------------------------
// HOST FUNCTIONS

void display_summed_weights(thrustDvec<float> sumsOfWeights) {
	std::cout << std::endl << "Sums of weights:" << std::endl;
	for(auto weight : sumsOfWeights)
		std::cout << weight << "\t";
}

void display_topographic_distances(thrustDvec<float> dist_bmu1_2, int numberToDisplay = 10) {
	std::cout << std::endl << "Topographic distance per article bmu1 to bmu2:" << std::endl;
	for(int i = 0; i < numberToDisplay; i++) 
		std::cout << i << "\t" << dist_bmu1_2[i] << "\t";
}

void display_distances_to_bmus(thrustDvec<float> distance1, thrustDvec<float> distance2, ulong numberOfArticles, ulong numberToDisplay = 10) {
	std::cout << std::endl << "Distance from articles to bmus:" << std::endl;
	for (int i = 0; i < numberToDisplay; i++) {
		std::cout << "Article " << i << ":\t" << distance1[i] << std::endl;
	}	for (int i = numberOfArticles - numberToDisplay - 1; i < numberOfArticles; i++) {
		std::cout << "Article " << i << ":\t" << distance1[i] << std::endl;
	}
}

void display_bmus(thrustDvec<float> bmu1, thrustDvec<float> bmu2, ulong numberOfArticles, ulong numberToDisplay = 10) {
	std::cout << std::endl << "BMUs for each article:" << std::endl;
	for (int i = 0; i < numberToDisplay; i++) {
		std::cout << "Article " << i << ":\t" << bmu1[i] << std::endl;
	}
	for (int i = numberOfArticles - numberToDisplay - 1; i < numberOfArticles; i++) {
		std::cout << "Article " << i << ":\t" << bmu1[i] << std::endl;
	}
	for (int i = 0; i < numberToDisplay; i++) {
		std::cout << "Article " << i << ":\t" << bmu2[i] << std::endl;
	}
	for (int i = numberOfArticles - numberToDisplay - 1; i < numberOfArticles; i++) {
		std::cout << "Article " << i << ":\t" << bmu2[i] << std::endl;
	}
}

void display_selected_bmus(thrustDvec<float> bmu1, std::vector<ulong> selected_index, std::vector<std::string> selected_pmid, ulong numberToDisplay = 10) {
	std::cout << std::endl << "BMUs for each article:" << std::endl;
	for (int i = 0; i < std::min(static_cast<size_t>(numberToDisplay), selected_index.size()); i++) {
		std::cout << "Article " << selected_pmid[i] << ":\t" << bmu1[selected_index[i]] << std::endl;
	}
}

void save_selected_bmus(thrustDvec<float> bmu1, std::vector<ulong> selected_index, std::vector<std::string> selected_pmid, int epoch)
{
	// file pointer
	std::fstream fout;

	// opens an existing csv file or creates a new file.
	fout.open("selected_pmids" + std::to_string(epoch) + ".csv", std::ios::out | std::ios::app);

	// export bmu information
	for (long i = 0; i < selected_index.size(); i++) {
		// Insert the data to file
		fout << selected_pmid[i] << ", "
			<< bmu1[selected_index[i]]
			<< "\n";
	}
	fout.close();
}

void display_numerators(thrustDvec<float> numerators) {
	for(auto numerator : numerators)
		std::cout << numerator << "\t";
}
