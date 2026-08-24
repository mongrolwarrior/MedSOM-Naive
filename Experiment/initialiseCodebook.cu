#include "initialiseCodebook.cuh"

int main()
{
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	std::cout << numSMs << std::endl;

	thrustDvec<float>	codebook_dev(codebookSize);
	thrustHvec<float>	codebook_host(codebookSize);

	std::random_device rd;
	long long seed = rd();

	thrustDvec<curandState> states(threads * blocks); // generator states

	cx::timer tim;   // start clock
	gpu_random_initialize<<<blocks, threads >>>(seed, state.data().get(), threads * blocks);
	cudaDeviceSynchronize();

	initializeCodebook<<<blocks, threads >>>(codebook_dev.data().get(), state.data().get(), codebook_dev.size());
	cudaDeviceSynchronize();
	codebook_host = codebook_dev;
	show_codebook_subset(codebook_host);

	double t1 = tim.lap_ms();

	//	DESCRIBE RESULTS
	std::cout << "Processing time: " << t1 << "ms" << std::endl;

	// REDUCE FOR MAX/MIN/AVE WEIGHTS
	float sumWeights = thrust::reduce(codebook_dev.begin(), codebook_dev.end());
	float minWeight = *(thrust::min_element(thrust::device, codebook_dev.begin(), codebook_dev.end()));
	float maxWeight = *(thrust::max_element(thrust::device, codebook_dev.begin(), codebook_dev.end()));
	std::cout << "Number of weights: " << codebook_dev.size() << "\tExpected: " << dimX * dimY * numberOfMeshCategories << std::endl;
	std::cout << "Ave value: " << sumWeights / codebookSize << "\tMin weight: " << minWeight << "\tMax weight: " << maxWeight << std::endl;

	return 0;
}
