
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int globalID = blockIdx.x*blockDim.x + threadIdx.x;

	if(globalID < size) {
		y[globalID] = x[globalID]*scale + y[globalID];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	float scale = 2.0f;

	int blockSize = 256;
	int gridSize = (vectorSize + blockSize -1)/blockSize;

	float *h_x = (float *) malloc(vectorSize * sizeof(float));
	float *h_y = (float *) malloc(vectorSize * sizeof(float));
	float *h_saxpy_result = (float *) malloc(vectorSize * sizeof(float));

	if (h_x == NULL || h_y == NULL || h_saxpy_result == NULL) {
		printf("Unable to malloc host memory ... Exiting!");
		return -1;
	}

	vectorInit(h_x, vectorSize);
	vectorInit(h_y, vectorSize);

	float *d_x = nullptr;
	float *d_y = nullptr;

	cudaMalloc(&d_x, vectorSize*sizeof(float));
	cudaMalloc(&d_y, vectorSize*sizeof(float));

	if (!d_x || !d_y) {
		printf("Unable to malloc device memory ... Exiting!");
		return -1;
	}

	cudaMemcpy(d_x, h_x, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" x = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_x[i]);
		}
		printf(" ... }\n");
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_y[i]);
		}
		printf(" ... }\n");
	#endif

	saxpy_gpu<<<gridSize, blockSize>>>(d_x, d_y, scale, vectorSize);
	cudaMemcpy(h_saxpy_result, d_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" saxpy_result = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", h_saxpy_result[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(h_x, h_y, h_saxpy_result, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	std::cout << "runGpuSaxpy complete\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
