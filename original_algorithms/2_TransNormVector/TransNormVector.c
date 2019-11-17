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

void trans_norm_vector(double* A, double* x, double* y, double* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++) {
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++) {
      		tmp[i] = 0;

	      	for (j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
	      	for (j = 0; j < NY; j++) {
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
	}
}

int main(int argc, char *argv[])
{
	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	struct timeval	cpu_start, cpu_end;

	A = (double*)malloc(NX*NY*sizeof(double));
	x = (double*)malloc(NY*sizeof(double));
	y = (double*)malloc(NY*sizeof(double));
	tmp = (double*)malloc(NX*sizeof(double));

	init_array(x, A);

	gettimeofday(&cpu_start, NULL);
	trans_norm_vector(A, x, y, tmp);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	free(A);
	free(x);
	free(y);
	free(tmp);

  	return 0;
}

