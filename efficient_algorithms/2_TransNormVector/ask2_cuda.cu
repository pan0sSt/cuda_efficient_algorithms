#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size. */
#define NX 4096
#define NY 4096

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(double *x, double *A)
{
	int i, j;

	for (i = 0; i < NX; i++) {
		// x[i] = i * M_PI; <-- ΝΧ επαναλήψεις - Λάθος
		for (j = 0; j < NY; j++) {
			A[i*NY + j] = ((double) i*(j)) / NX;
		}
	}

	for (j = 0; j < NY; j++) {
		x[j] = j * M_PI;
	}
}

__global__ void trans_norm_vector_kernel1(double* A, double* x, double* tmp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	__shared__ double tmp_shared;

	if (i < NX)
	{
		tmp_shared = tmp[i];
	 	for(int j = 0; j < NY; j++)
	 	{
	 		tmp_shared = tmp_shared + A[i*NY+j] * x[j];
	 	}
	 	tmp[i] = tmp_shared;
	}
}

__global__ void trans_norm_vector_kernel2(double* A, double* y, double* tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ double y_shared;
	
	if (j < NY)
	{
		y_shared = y[j];
		for(int i = 0; i < NX; i++)
		{
			y_shared = y_shared + A[i*NY+j] * tmp[i];
		}
		y[j] = y_shared;
	}
}

int main(int argc, char *argv[])
{
	double		*A_h, *A_d;
	double		*x_h, *x_d;
	double		*y_h, *y_d;
	double		*tmp_h, *tmp_d;
	struct timeval	cpu_start, cpu_end;

	A_h = (double*)malloc(NX*NY*sizeof(double));
	x_h = (double*)malloc(NY*sizeof(double));
	y_h = (double*)malloc(NY*sizeof(double));
	tmp_h = (double*)malloc(NX*sizeof(double));

	// Δέσμευση μνήμης στο device για τα διανύσματα
	cudaMalloc((void **) &A_d, NX*NY*sizeof(double));
	cudaMalloc((void **) &x_d, NY*sizeof(double));
	cudaMalloc((void **) &y_d, NY*sizeof(double));
	cudaMalloc((void **) &tmp_d, NX*sizeof(double));

	cudaMemset(y_d, 0, NY*sizeof(double));
	cudaMemset(tmp_d, 0, NX*sizeof(double));

	init_array(x_h, A_h);

	// Αντιγραφή x στο device
	cudaMemcpy(x_d, x_h, NY*sizeof(double), cudaMemcpyHostToDevice);
	// Αντιγραφή A στο device
	cudaMemcpy(A_d, A_h, NX*NY*sizeof(double), cudaMemcpyHostToDevice);

	
	//----------------------------------------------------------------------

	// Κάθε block θα έχει διάσταση 16x16
	unsigned int BLOCK_SIZE_PER_DIM = 16;

	// Στρογγυλοποίηση προς τα πάνω για το πλήθος των block σε κάθε διάσταση
	unsigned int numBlocksX = (NX - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (NY - 1) / BLOCK_SIZE_PER_DIM + 1;

	// Ορισμός διαστάσεων πλέγματος
	dim3 dimGrid1(numBlocksX, 1);
	dim3 dimGrid2(numBlocksY, 1);

	// Ορισμός διαστάσεων block
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);


	//----------------------------------------------------------------------

	gettimeofday(&cpu_start, NULL);

	// Κλήση υπολογιστικού πυρήνα
	trans_norm_vector_kernel1<<<dimGrid1, dimBlock>>>(A_d, x_d, tmp_d);
	cudaThreadSynchronize();
	trans_norm_vector_kernel2<<<dimGrid2, dimBlock>>>(A_d, y_d, tmp_d);
	cudaThreadSynchronize();


	cudaMemcpy(y_h, y_d, NY*sizeof(double), cudaMemcpyDeviceToHost);

	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "GPU Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	printf("================================\n");

	FILE *f = fopen("ask2_cuda_output.txt", "w+");
	if (f == NULL)
	{
	    printf("Error opening ask2_cuda_output.txt!\n");
	    exit(1);
	}

	for (int i = 0; i < NY; i++) {
			fprintf(f, "%f\n", y_h[i]);
	}

	if(f) { printf("Results saved in ask2_cuda_output.txt!\n"); }

	fclose(f);



	// Αποδέσμευση μνήμης στον host
	free(A_h);
	free(x_h);
	free(y_h);
	free(tmp_h);

	// Αποδέσμευση μνήμης στον host
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(tmp_d);

  	return 0;
}

