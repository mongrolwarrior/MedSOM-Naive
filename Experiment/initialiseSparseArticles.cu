#include "initialiseSparseArticles.cuh"

int main()
{
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	// DEVICE VECTORS
	thrustDvec<ulong>	nnz_per_article(numberOfArticles);
	thrustDvec<ulong>	article_indices(numberOfArticles);
	thrustDvec<ulong>	rows;
	thrustDvec<ulong>	columns;
	// HOST VECTORS
	thrustHvec<ulong>	nnz_per_article_host(numberOfArticles);

	std::random_device rd;
	long long seed = rd();

	thrustDvec<curandState> states(threads * blocks); // generator states

	cx::timer tim;   // start clock
	initialise_random_states<<<blocks, threads>>>(seed, states.data().get(), threads * blocks);
	cudaDeviceSynchronize();

	get_random_nnz<<<blocks, threads>>>(nnz_per_article.data().get(), states.data().get(), numberOfArticles);
	cudaDeviceSynchronize();
	ulong number_of_nnz_elements = thrust::reduce(nnz_per_article.begin(), nnz_per_article.end());
	rows.resize(number_of_nnz_elements);
	columns.resize(number_of_nnz_elements);

	nnz_per_article_host = nnz_per_article;
	display_nnz_subset(nnz_per_article_host, 10);

	thrust::exclusive_scan(nnz_per_article.begin(), nnz_per_article.end(), article_indices.begin());
	std::cout << "Number of nnz: " << number_of_nnz_elements << "\tScan last: " << article_indices[numberOfArticles - 1] <<
		"\tNnz last article: " << nnz_per_article[numberOfArticles - 1] << std::endl;
	initialise_sparse_articles<<<blocks, threads>>>(rows.data().get(), columns.data().get(), nnz_per_article.data().get(), article_indices.data().get(), states.data().get(), numberOfArticles, numberOfMeshCategories);

	display_sparse_articles(article_indices, columns, nnz_per_article);

	double t1 = tim.lap_ms();
}