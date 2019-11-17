#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define M 1024
#define N 1024


#define FLOAT_N 3214212.01

void init_arrays(double* data)
{
	int i, j;

	for (i = 1; i < (M+1); i++) {
		for (j = 1; j < (N+1); j++) {
			data[i*(N+1) + j] = ((double) i*j) / M;
		}
	}
}

__global__ void mean_kernel(double* data, double* mean)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	__shared__ double mean_shared;

	if (j < (M+1))
	{
		mean_shared = 0.0;

		int i;
		for(i = 1; i < (N+1); i++)
		{
			mean_shared += data[i * (M+1) + j];
		}
		mean_shared /= FLOAT_N;
		mean[j] = mean_shared;
	}

}

__global__ void data_kernel(double* data, double* mean)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		
	if ((i < (N+1)) && (j < (M+1)))
	{
		data[i * (M+1) + j] -= mean[j];	
	}
}

__global__ void symmat_kernel(double* symmat, double* data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i, j2;
	__shared__ double symmat_shared;

	if (j1 < (M+1))
	{
		for (j2 = j1; j2 < (M+1); j2++)
		{		
			symmat_shared = 0.0;
			for(i = 1; i < (N+1); i++)
			{
				symmat_shared += data[i * (M+1) + j1] * data[i * (M+1) + j2];
			}
			symmat[j1 * (M+1) + j2] = symmat_shared;
			symmat[j2 * (M+1) + j1] = symmat[j1 * (M+1) + j2];
		}
	}
}

int main(int argc, char *argv[])
{
	double		*data_h, *data_d;
	double		*symmat_h, *symmat_d;
	double		*mean_h, *mean_d;
	struct timeval	cpu_start, cpu_end;

	data_h = (double*)malloc((M+1)*(N+1)*sizeof(double));
	mean_h = (double*)malloc((M+1)*sizeof(double));
	symmat_h = (double*)malloc((M+1)*(M+1)*sizeof(double));

	// Δέσμευση μνήμης στο device για τα διανύσματα
	cudaMalloc((void **) &data_d, (M+1)*(N+1)*sizeof(double));
	cudaMalloc((void **) &mean_d, (M+1)*sizeof(double));
	cudaMalloc((void **) &symmat_d, (M+1)*(M+1)*sizeof(double));

	cudaMemset(data_d, 0, (M+1)*(N+1)*sizeof(double));
	// cudaMemset(symmat_d, 0, (M+1)*(M+1)*sizeof(double));
	// cudaMemset(mean_d, 0, (M+1)*sizeof(double));

	init_arrays(data_h);

	// Αντιγραφή data στο device
	cudaMemcpy(data_d, data_h, (M+1)*(N+1)*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(symmat_d, symmat_h, (M+1)*(M+1)*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(mean_d, mean_h, (M+1)*sizeof(double), cudaMemcpyHostToDevice);

	
	//----------------------------------------------------------------------

	// Κάθε block θα έχει διάσταση 16x16
	unsigned int BLOCK_SIZE_PER_DIM = 16;

	// Ορισμός διαστάσεων πλέγματος
	dim3 dimGrid1((M - 1) / BLOCK_SIZE_PER_DIM + 1, 1);
	dim3 dimGrid2((M - 1) / BLOCK_SIZE_PER_DIM + 1, (N - 1) / BLOCK_SIZE_PER_DIM + 1);
	dim3 dimGrid3((M - 1) / BLOCK_SIZE_PER_DIM + 1, 1);

	// Ορισμός διαστάσεων block
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);




	//----------------------------------------------------------------------

	gettimeofday(&cpu_start, NULL);
	
	mean_kernel<<<dimGrid1, dimBlock>>>(data_d, mean_d);
	cudaThreadSynchronize();
	data_kernel<<<dimGrid2, dimBlock>>>(data_d, mean_d);
	cudaThreadSynchronize();
	symmat_kernel<<<dimGrid3, dimBlock>>>(symmat_d,data_d);
	cudaThreadSynchronize();

	cudaMemcpy(symmat_h, symmat_d, (M+1)*(M+1)*sizeof(double), cudaMemcpyDeviceToHost);


	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);
	printf("================================\n");

	FILE *f = fopen("ask3_cuda_output.txt", "w+");
	if (f == NULL)
	{
	    printf("Error opening ask3_cuda_output.txt!\n");
	    exit(1);
	}


	for(int i = 1; i < (M+1); i++) {
		for(int j = 1; j < (M+1); j++){
			fprintf(f, "%f\n", symmat_h[i * (M+1) + j]);
		}
	}

	if(f) { printf("Results saved in ask3_cuda_output.txt!\n"); }

	fclose(f);

	// Αποδέσμευση μνήμης στον host
	free(data_h);
	free(mean_h);
	free(symmat_h);

	// Αποδέσμευση μνήμης στον host
	cudaFree(data_d);
	cudaFree(mean_d);
	cudaFree(symmat_d);

  	return 0;
}

