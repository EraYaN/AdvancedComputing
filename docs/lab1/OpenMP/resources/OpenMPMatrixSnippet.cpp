void matrix_mult_pl(int size, user_float_t *matrix1_in, user_float_t *matrix2_in, user_float_t *matrix_out) {
    int rowsOut, rowsIn, cols;
    int j;
# pragma omp parallel               \
    shared(size, matrix1_in, matrix2_in, matrix_out)    \
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