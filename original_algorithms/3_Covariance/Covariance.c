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

void covariance(double* data, double* symmat, double* mean)
{
	int	i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j < (M+1); j++) {
		mean[j] = 0.0;
		for (i = 1; i < (N+1); i++) {
        		mean[j] += data[i*(M+1) + j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 1; i < (N+1); i++) {
		for (j = 1; j < (M+1); j++) {
			data[i*(M+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 < (M+1); j1++) {
		for (j2 = j1; j2 < (M+1); j2++) {
	       		symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i < N+1; i++) {
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
        		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
      		}
	}
}

int main(int argc, char *argv[])
{
	double		*data;
	double		*symmat;
	double		*mean;
	struct timeval	cpu_start, cpu_end;

	data = (double*)malloc((M+1)*(N+1)*sizeof(double));
	symmat = (double*)malloc((M+1)*(M+1)*sizeof(double));
	mean = (double*)malloc((M+1)*sizeof(double));

	init_arrays(data);

	gettimeofday(&cpu_start, NULL);
	covariance(data, symmat, mean);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	free(data);
	free(symmat);
	free(mean);

  	return 0;
}

