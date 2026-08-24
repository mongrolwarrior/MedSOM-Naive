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
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <functional>
#include "initialiseCodebook.cuh"
#include "initialiseSparseArticles.cuh"
#include "bmus.cuh"
#include <sstream>
#include <cstring>

struct ssTopDistSS_functor
{
	const float a;

	ssTopDistSS_functor(float _a) : a(_a) {}

	__host__ __device__
		float operator()(const float& x) const {
		return (a - x) * (a - x);
	}
};

std::vector<ulong> compute_order(std::vector<float> v)
{
	std::vector<ulong> indices(v.size());
	std::iota(indices.begin(), indices.end(), 0u);
	std::stable_sort(indices.begin(), indices.end(), [&](ulong lhs, ulong rhs) {
		return v[lhs] < v[rhs];
	});
	std::vector<ulong> res(v.size());
	for (ulong i = 0; i != indices.size(); ++i) {
		res[indices[i]] = i;
	}
	return res;
}

// ---------------------------------------------------------------------------
// .sbcsr loader — reads the SparseBinarySOM corpus format directly
// ---------------------------------------------------------------------------
struct SbcsrCorpus {
	uint32_t n_samples;
	uint32_t n_features;
	uint32_t n_nonzeros;
	std::vector<uint32_t> row_ptr;   // [n_samples + 1]
	std::vector<uint16_t> col_idx;   // [n_nonzeros]
};

SbcsrCorpus load_sbcsr(const char* path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) { std::cerr << "Cannot open " << path << std::endl; exit(1); }

	char magic[8];
	f.read(magic, 8);
	if (std::memcmp(magic, "SBCSR1\0\0", 8) != 0 &&
	    std::memcmp(magic, "SCSR1\0\0\0", 8) != 0) {
		std::cerr << "Bad .sbcsr magic in " << path << std::endl;
		exit(1);
	}

	SbcsrCorpus c;
	uint32_t has_values;
	f.read(reinterpret_cast<char*>(&c.n_samples),  4);
	f.read(reinterpret_cast<char*>(&c.n_features), 4);
	f.read(reinterpret_cast<char*>(&c.n_nonzeros), 4);
	f.read(reinterpret_cast<char*>(&has_values),    4);

	c.row_ptr.resize(c.n_samples + 1);
	f.read(reinterpret_cast<char*>(c.row_ptr.data()), (c.n_samples + 1) * sizeof(uint32_t));

	c.col_idx.resize(c.n_nonzeros);
	f.read(reinterpret_cast<char*>(c.col_idx.data()), c.n_nonzeros * sizeof(uint16_t));

	std::cout << "Loaded " << path << ": " << c.n_samples << " samples, "
	          << c.n_features << " features, " << c.n_nonzeros << " nnz" << std::endl;
	return c;
}

void print_usage(const char* prog) {
	std::cerr << "Usage (sbcsr): " << prog << " <corpus.sbcsr> [epochs] [dimX] [dimY]" << std::endl;
	std::cerr << "Usage (bin):   " << prog << " <bin_dir> <mesh_xml> [article_filter] [epochs] [dimX] [dimY]" << std::endl;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
	return s.size() >= suffix.size() &&
	       s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		print_usage(argv[0]);
		return 1;
	}

	bool sbcsr_mode = ends_with(argv[1], ".sbcsr") || ends_with(argv[1], ".scsr");

	// ---------------------------------------------------------------------------
	// Parse CLI and load data
	// ---------------------------------------------------------------------------
	thrustDvec<ulong> nnz_per_article;
	thrustDvec<ulong> article_indices;
	thrustDvec<ulong> columns;
	ulong total_nnz = 0;

	if (sbcsr_mode) {
		// .sbcsr path — no Boost, no MeSH XML
		if (argc > 2) number_of_epochs = std::stoul(argv[2]);
		if (argc > 3) {
			dimX = std::stoul(argv[3]);
			dimY = (argc > 4) ? std::stoul(argv[4]) : dimX;
			radius_start = std::min(dimX, dimY) / 2;
		}

		SbcsrCorpus corpus = load_sbcsr(argv[1]);
		numberOfMeshCategories = corpus.n_features;
		numberOfArticles = corpus.n_samples;
		total_nnz = corpus.n_nonzeros;

		// Build host vectors in medsom's format (ulong)
		thrust::host_vector<ulong> h_nnz(numberOfArticles);
		thrust::host_vector<ulong> h_indices(numberOfArticles);
		thrust::host_vector<ulong> h_columns(total_nnz);

		for (uint32_t i = 0; i < numberOfArticles; i++) {
			h_nnz[i] = corpus.row_ptr[i + 1] - corpus.row_ptr[i];
			h_indices[i] = corpus.row_ptr[i];
		}
		for (uint32_t i = 0; i < total_nnz; i++) {
			h_columns[i] = corpus.col_idx[i];
		}

		nnz_per_article = h_nnz;
		article_indices = h_indices;
		columns = h_columns;

	} else {
		std::cerr << "This build supports only .sbcsr/.scsr corpora (legacy .bin/MeSH path removed for the public release).\n";
		return 2;
	}

	// ---------------------------------------------------------------------------
	// Common setup
	// ---------------------------------------------------------------------------
	codebookSize = dimX * dimY * numberOfMeshCategories;

	std::cout << std::endl;
	std::cout << "SOM grid:        " << dimX << "x" << dimY << std::endl;
	std::cout << "Epochs:          " << number_of_epochs << std::endl;
	std::cout << "Articles:        " << numberOfArticles << std::endl;
	std::cout << "MeSH categories: " << numberOfMeshCategories << std::endl;
	std::cout << "Codebook size:   " << codebookSize << " floats ("
	          << (codebookSize * sizeof(float)) / (1024ULL * 1024) << " MiB)" << std::endl;

	// Training vectors
	thrustDvec<float> sumsOfWeights(dimX * dimY);
	thrustDvec<float> distance_article_to_bmu1(numberOfArticles);
	thrustDvec<float> distance_article_to_bmu2(numberOfArticles);
	thrustDvec<ulong> bmu1(numberOfArticles);
	thrustDvec<ulong> bmu2(numberOfArticles);
	thrustDvec<float> dist_bmu1_2(numberOfArticles);
	thrustDvec<float> neighborhood_function(numberOfArticles);
	thrustDvec<float> numerators(numberOfMeshCategories);

	thrustDvec<float>	codebook(codebookSize);
	thrustHvec<float>   h_codebook(codebookSize);

	// ---------------------------------------------------------------------------
	// Random codebook init
	// ---------------------------------------------------------------------------
	std::random_device rd;
	long long seed = rd();

	thrustDvec<curandState> states(threads * blocks);

	gpu_random_initialize<<<blocks, threads>>>(seed, states.data().get(), threads * blocks);
	cudaDeviceSynchronize();
	initializeCodebook<<<blocks, threads>>>(codebook.data().get(), states.data().get(), codebook.size());
	cudaDeviceSynchronize();

	std::cout << "Codebook initialized randomly" << std::endl;

	// ---------------------------------------------------------------------------
	// Training loop
	// ---------------------------------------------------------------------------
	cx::timer tim;
	for (int epoch = 0; epoch < (int)number_of_epochs; epoch++) {
		float radius = std::max(1.0, (double)radius_start / std::pow(1.7, epoch));

		std::cout << "\nEpoch " << epoch << "/" << number_of_epochs
		          << "  radius=" << radius << std::endl;

		thrust::fill(sumsOfWeights.begin(), sumsOfWeights.end(), 0.0f);
		sumWeights<<<blocks, threads>>>(sumsOfWeights.data().get(), codebook.data().get(),
			numberOfMeshCategories, dimX, dimY);
		cudaDeviceSynchronize();

		calculate_distances_articles_to_bmus<<<blocks, threads>>>(
			distance_article_to_bmu1.data().get(), distance_article_to_bmu2.data().get(),
			bmu1.data().get(), bmu2.data().get(),
			sumsOfWeights.data().get(), codebook.data().get(),
			nnz_per_article.data().get(), columns.data().get(), article_indices.data().get(),
			dimX, dimY, numberOfMeshCategories, numberOfArticles);
		cudaDeviceSynchronize();

		calculateTopographicError<<<blocks, threads>>>(bmu1.data().get(), bmu2.data().get(),
			dist_bmu1_2.data().get(), dimX, dimY, numberOfArticles);
		cudaDeviceSynchronize();
		float topographicAveDistance = thrust::reduce(dist_bmu1_2.begin(), dist_bmu1_2.end()) / numberOfArticles;
		float topographicError = (float)thrust::count_if(thrust::device, dist_bmu1_2.begin(),
			dist_bmu1_2.end(), is_greater_than_1()) / (float)numberOfArticles;

		float qe = thrust::reduce(distance_article_to_bmu1.begin(),
			distance_article_to_bmu1.end()) / numberOfArticles;

		std::cout << "  QE: " << qe
		          << "  TE: " << topographicError
		          << "  Ave topo dist: " << topographicAveDistance << std::endl;

		for (ulong node = 0; node < dimX * dimY; node++) {
			neighborhoodFnForNode<<<blocks, threads>>>(codebook.data().get(),
				bmu1.data().get(), neighborhood_function.data().get(),
				dimX, dimY, node, numberOfArticles, radius, std_coeff);
			cudaDeviceSynchronize();
			float denominator = thrust::reduce(neighborhood_function.begin(),
				neighborhood_function.end());

			calculateNumeratorsForNode<<<blocks, threads>>>(numerators.data().get(),
				neighborhood_function.data().get(), nnz_per_article.data().get(),
				article_indices.data().get(), columns.data().get(),
				numberOfMeshCategories, numberOfArticles);
			cudaDeviceSynchronize();

			calculateNewWeights<<<blocks, threads>>>(codebook.data().get(),
				numerators.data().get(),
				dimX, dimY, numberOfMeshCategories, node, denominator);
			cudaDeviceSynchronize();

			thrust::fill(numerators.begin(), numerators.end(), 0);
		}
		double t1 = tim.lap_ms();
		printf("  Epoch %d time: %.3f min\n", epoch, t1 / 60000);

		if (topographicError < 0.4) {
			h_codebook = codebook;
			std::string fname = "codebook_epoch" + std::to_string(epoch) + ".dat";
			std::ofstream fout(fname, std::ios::out | std::ios::binary);
			fout.write(reinterpret_cast<char const*>(h_codebook.data()),
				h_codebook.size() * sizeof(float));
			fout.close();
			std::cout << "  Saved checkpoint: " << fname << std::endl;
		}
	}

	// Save final codebook
	h_codebook = codebook;
	{
		std::string fname = "codebook_final.dat";
		std::ofstream fout(fname, std::ios::out | std::ios::binary);
		fout.write(reinterpret_cast<char const*>(h_codebook.data()),
			h_codebook.size() * sizeof(float));
		fout.close();
		std::cout << "\nSaved: " << fname << std::endl;
	}

	std::atexit([] {cudaDeviceReset(); });
	return 0;
}
