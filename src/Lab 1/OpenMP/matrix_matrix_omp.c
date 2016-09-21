#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// generate two input matrices
void matrix_matrix_gen(int size, double *matrix1, double *matrix2) {
	int i;
	for (i = 0; i < size*size; i++)
		matrix1[i] = ((double)rand()) / 5307.0;
	for (i = 0; i < size*size; i++)
		matrix2[i] = ((double)rand()) / 5307.0;
}

// matrix matrix multiplication
void matrix_matrix_mult_sq(int size, double *matrix1_in,
	double *matrix2_in, double *matrix_out) {
	int rowsOut, rowsIn, cols;
	int j;
	for (cols = 0; cols<size; cols++) {
		for (rowsOut = 0; rowsOut<size; rowsOut++) {
			matrix_out[cols + rowsOut*size] = 0.0;
			for (j = 0, rowsIn = 0; rowsIn<size; j++, rowsIn++) {
				matrix_out[cols + rowsOut*size] += matrix1_in[j + rowsOut*size] * matrix2_in[rowsIn*size + cols];
			}
		}
	}
}

void matrix_matrix_mult_pl(int size, double *matrix1_in,
	double *matrix2_in, double *matrix_out) {
	int rowsOut, rowsIn, cols;
	int j;
# pragma omp parallel				\
	shared(size, matrix1_in, matrix2_in, matrix_out)	\
	private(rowsOut, rowsIn, cols, j)
# pragma omp for
	for (cols = 0; cols<size; cols++) {
# pragma omp for
		for (rowsOut = 0; rowsOut<size; rowsOut++) {
			matrix_out[cols + rowsOut*size] = 0.0;
			for (j = 0, rowsIn = 0; rowsIn<size; j++, rowsIn++) {
				matrix_out[cols + rowsOut*size] += matrix1_in[j + rowsOut*size] * matrix2_in[rowsIn*size + cols];
			}
		}
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
	double *matrix1 = (double *)malloc(sizeof(double)*size*size);
	double *matrix2 = (double *)malloc(sizeof(double)*size*size);
	double *result_sq = (double *)malloc(sizeof(double)*size*size);
	double *result_pl = (double *)malloc(sizeof(double)*size*size);
	matrix_matrix_gen(size, matrix1, matrix2);

	double time_sq = 0;
	double time_pl = 0;

	omp_set_num_threads(threads);
	time_sq = omp_get_wtime();
	matrix_matrix_mult_sq(size, matrix1, matrix2, result_sq);
	time_sq = omp_get_wtime() - time_sq;

	time_pl = omp_get_wtime();
	matrix_matrix_mult_pl(size, matrix1, matrix2, result_pl);
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

	free(matrix1);
	free(matrix2);
	free(result_sq);
	free(result_pl);

	getchar();

	printf("Done.");
	return 0;
}
