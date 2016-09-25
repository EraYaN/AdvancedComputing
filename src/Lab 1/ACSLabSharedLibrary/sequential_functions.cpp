#include "sequential_functions.h"

void mv_mult_sq(int size, user_float_t *vector_in, user_float_t *matrix_in, user_float_t *vector_out) {
	for (int cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (int j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

// matrix matrix multiplication
void mm_mult_sq(int size, user_float_t *matrix1_in, user_float_t *matrix2_in, user_float_t *matrix_out) {
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

void matrix_gen(int size, user_float_t *matrix) {
	for (int i = 0; i < size*size; i++)
		matrix[i] = ((user_float_t)rand()) / 5307.0;
}

void vector_gen(int size, user_float_t *vector) {
	for (int i = 0; i < size; i++)
		vector[i] = ((user_float_t)rand()) / 65535.0;
}

void matrix_gen(int paddedSize, int size, user_float_t *matrix) {
	for (int i = 0; i < paddedSize*paddedSize; i++)
		if (i % paddedSize < size && i < paddedSize * size) {
			matrix[i] = ((user_float_t)rand()) / 5307.0;
		} else {
			matrix[i] = 0.0;
		}
}

void vector_gen(int size, int sizeReal, user_float_t *vector) {
	for (int i = 0; i < size; i++) {
		if (i < sizeReal) {
			vector[i] = i*1.2f + 1;//((float)rand())/65535.0f;
		} else {
			vector[i] = 0.0;
		}
	}
}