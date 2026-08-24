#include <cuda_runtime_api.h>
#include <iostream>
#include "compat.h"
#include "curand_kernel.h"
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <fstream>
// CURRENT PROJECT
#include "parameters.cuh"

//--------------------------------------------------------------------------
// DEVICE FUNCTIONS

__global__ void initialise_random_states(long long seed, curandState* states, ulong total_threads) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= total_threads) return;

	curand_init(seed + id, 0, 0, &states[id]);
}

__global__ void get_random_nnz(ulong* nnz, curandState* states, ulong numberOfArticles, int minMeshPerArticle, int maxMeshPerArticle) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	curandState localState = states[id];

	for (ulong i = id; i < numberOfArticles; i += stride) {
		nnz[i] = curand(&localState) % (maxMeshPerArticle - minMeshPerArticle + 1) + minMeshPerArticle;
	}
	states[id] = localState;
}

__global__ void initialise_sparse_articles(ulong* rows, ulong* columns, ulong* nnz, ulong* article_indices, curandState* states, ulong numberOfArticles, ulong numberOfMeshCategories) {
	ulong id = threadIdx.x + blockDim.x * blockIdx.x;
	ulong stride = blockDim.x * gridDim.x;

	if (id >= numberOfArticles) return;

	curandState localState = states[id];

	for (ulong article = id; article < numberOfArticles; article += stride) {
		for(ulong j = 0; j < nnz[article]; j++) {
			// TO PARALLELISE WITH RANDOM NUMBER OF MESH PER ARTICLES, NEEDED TO PUT INDICES OF FIRST MESH FOR ARTICLE IN article_indices VECTOR
			// ACTUAL INDEX IS THEN FIRST INDEX (article_indices[id]) PLUS MESH NUMBER OF THIS ARTICLE (j)
			ulong base_index = article_indices[article];
			rows[base_index + j] = article;
			bool match = true;
			while (match != false)
			{
				match = false;
				ulong temp_mesh_index = curand(&localState) % numberOfMeshCategories;
				columns[base_index + j] = temp_mesh_index;
				if(j != 0)
					for (ulong nnz_index = 0; nnz_index < j; nnz_index++) {
						if(columns[base_index + nnz_index] == temp_mesh_index) 
							match = true;
					}						
			}			
		}
	}
	states[id] = localState;
}


//--------------------------------------------------------------------------
// HOST FUNCTIONS

void display_nnz_subset(thrustHvec<ulong> nnz, ulong number_to_show = 5) {
	std::cout << std::endl << "Number of non-zero articles in each article:\t";
	for (int i = 0; i < number_to_show; i++)
		std::cout << nnz[i] << "\t";
	for (int i = nnz.size() - number_to_show; i < nnz.size(); i++)
		std::cout << nnz[i] << "\t";
	std::cout << std::endl << std::endl;
}

void display_article_indices(thrustHvec<float> indices, ulong number_to_show = 5) {
	std::cout << std::endl << "Article indices beginning: " << std::endl;
	for (int i = 0; i < number_to_show; i++)
		std::cout << indices[i] << "\t";
	std::cout << std::endl << "Article indices end: " << std::endl;
	for (int i = numberOfArticles - number_to_show; i < numberOfArticles; i++)
		std::cout << indices[i] << "\t";
}

void display_sparse_articles(thrustHvec<ulong> article_indices, thrustHvec<ulong> columns, thrustHvec<ulong> nnz, int display_limit = 10) {
	for (int i = 0; i < display_limit; i++) {
		std::cout << "Row " << i << ":\t";
		for (int j = 0; j < nnz[i]; j++)
			if (j < display_limit) std::cout << columns[article_indices[i] + j] << "\t";
		std::cout << std::endl;
	}
	int countdown = 10;
	unsigned int row_index = article_indices.size() - 1;
	while (countdown > 0) {
		if (nnz[row_index] > 0) {
			std::cout << "Row " << row_index << ":\t";
			for (int j = 0; j < nnz[row_index]; j++)
				if (j < display_limit) std::cout << columns[article_indices[row_index] + j] << "\t";
			std::cout << std::endl;
			countdown--;
			row_index--;
		} else
			row_index--;
	}
}

void save_articles_dense(thrustDvec<ulong> rows, thrustDvec<ulong> columns, thrustDvec<ulong> nnz, ulong numberOfArticles) {
	std::ofstream fout("C:\\Users\\Andrew\\Downloads\\articles.txt");
	fout << std::setprecision(5);

	int nnz_index = 0;
	for (int article = 0; article < numberOfArticles; article++) {
		std::vector<ulong> this_row(numberOfMeshCategories);
		for (int mesh_index = 0; mesh_index < nnz[article]; mesh_index++) {
			this_row[columns[nnz_index]] = 1;
			nnz_index++;
		}
		for(int mesh_index = 0; mesh_index < numberOfMeshCategories; mesh_index++)
			fout << this_row[mesh_index] << "\t";
		fout << std::endl;
	}

	fout.close();
}