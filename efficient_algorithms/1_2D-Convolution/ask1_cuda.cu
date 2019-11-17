#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
// CUDA runtime
#include <cuda_runtime.h>

/* Problem size */
#define NI 4096
#define NJ 4096

__global__ void Convolution(double* A, double* B)
{
	int i, j;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	i = blockIdx.y*blockDim.y + threadIdx.y + 1;
    j = blockIdx.x*blockDim.x + threadIdx.x + 1;
    if (i < NI - 1 && j < NJ - 1) {
        B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		
    }
}

void init(double* A)
{
	int i, j;

	for (i = 0; i < NI; ++i) {
		for (j = 0; j < NJ; ++j) {
			A[i*NJ + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{
	double		*A_h, *B_h;
	double		*A_d, *B_d;
	struct timeval	cpu_start, cpu_end;

	
	A_h = (double*)malloc(NI*NJ*sizeof(double));
	B_h = (double*)malloc(NI*NJ*sizeof(double));

	// Δέσμευση μνήμης στο device για τα διανύσματα
	cudaMalloc((void **) &A_d, NI*NJ*sizeof(double));
	cudaMalloc((void **) &B_d, NI*NJ*sizeof(double));

	//initialize the arrays
	init(A_h);

	// Αντιγραφή A στο device
	cudaMemcpy(A_d, A_h, NI*NJ*sizeof(double), cudaMemcpyHostToDevice);
	
	//----------------------------------------------------------------------

	// Κάθε block θα έχει διάσταση 32×32
	unsigned int BLOCK_SIZE_PER_DIM = 32;

	// Στρογγυλοποίηση προς τα πάνω για το πλήθος των block σε κάθε διάσταση
	unsigned int numBlocksX = (NI - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (NJ - 1) / BLOCK_SIZE_PER_DIM + 1;

	// Ορισμός διαστάσεων πλέγματος
	dim3 dimGrid(numBlocksX, numBlocksY, 1);

	// Ορισμός διαστάσεων block
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);

	//----------------------------------------------------------------------
	
	gettimeofday(&cpu_start, NULL);
	
	// Κλήση υπολογιστικού πυρήνα
	Convolution<<<dimGrid, dimBlock>>>(A_d, B_d);

	cudaMemcpy(B_h, B_d, NI*NJ*sizeof(double), cudaMemcpyDeviceToHost);

	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// Αντιγραφή του αποτελέσματος στον host

	printf("================================\n");

	FILE *f = fopen("ask1_cuda_output.txt", "w+");
	if (f == NULL)
	{
	    printf("Error opening ask1_cuda_output.txt!\n");
	    exit(1);
	}

	for (int i = 1; i < NI - 1; ++i) {
		for (int j = 1; j < NJ - 1; ++j) {
			fprintf(f, "%f\n", B_h[i*NJ + j]);
			}
	}

	if(f) { printf("Results saved in ask1_cuda_output.txt!\n"); }

	fclose(f);

	// Αποδέσμευση μνήμης στον host
	free(A_h);
	free(B_h);
	
	// Αποδέσμευση μνήμης στο device
	cudaFree(A_d);
	cudaFree(B_d);
	
	return 0;
}
