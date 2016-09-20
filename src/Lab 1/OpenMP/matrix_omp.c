#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, double *matrix, double *vector) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = ((double)rand()) / 65535.0;
	for (i = 0; i < size*size; i++)
		matrix[i] = ((double)rand()) / 5307.0;
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, double *vector_in,
	double *matrix_in, double *vector_out) {
	int rows, cols;
	int j;
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

void matrix_mult_pl(int size, double *vector_in,
	double *matrix_in, double *vector_out) {
	int rows, cols;
	int j;
# pragma omp parallel				\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j)
# pragma omp for
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Got %d arguments, needs 3.", argc);
		printf("Usage: %s threads matrix/vector_size\n", argv[0]);
		return 1;
	}

	int size = atoi(argv[2]);
	int threads = atoi(argv[1]);
	double *vector = (double *)malloc(sizeof(double)*size);
	double *matrix = (double *)malloc(sizeof(double)*size*size);
	double *result_sq = (double *)malloc(sizeof(double)*size);
	double *result_pl = (double *)malloc(sizeof(double)*size);
	matrix_vector_gen(size, matrix, vector);

	double time_sq = 0;
	double time_pl = 0;

	omp_set_num_threads(threads);
	time_sq = omp_get_wtime();
	matrix_mult_sq(size, vector, matrix, result_sq);
	time_sq = omp_get_wtime() - time_sq;

	time_pl = omp_get_wtime();
	matrix_mult_pl(size, vector, matrix, result_pl);
	time_pl = omp_get_wtime() - time_pl;

	printf("DAT:%d\n", size);
	printf("THD:%d\n", omp_get_max_threads());
	printf("PRC:%d\n", omp_get_num_procs());
	printf("SEQ:%.14f\n", time_sq);
	printf("PAR:%.14f\n", time_pl);

	//check
	int i;
	for (i = 0; i < size; i++)
		if (result_sq[i] != result_pl[i]) {
			printf("wrong at position %d\n", i);
			return 2;
		}

	free(vector);
	free(matrix);
	free(result_sq);
	free(result_pl);

	printf("Done.");
	return 0;
}
