void matrix_mult_sse(int size, float *matrix1, float *matrix2, float *matrix_out) {
    __m128 a_line, b_line, r_line;
    int i, j, k;
# pragma omp parallel               \
    shared(size, matrix1, matrix2, matrix_out)  \
    private(i,j,k, a_line, b_line, r_line)
# pragma omp for
    for (k = 0; k < size * size; k += size) {
# pragma omp for
        for (i = 0; i < size; i += 4) {
            j = 0;
            b_line = _mm_load_ps(&matrix2[i]); //_mm_loadu_ps is the non-aligned version that only get a penalty with unaligned memory
            a_line = _mm_set1_ps(matrix1[j + k]);
            r_line = _mm_mul_ps(a_line, b_line);
            for (j = 1; j < size; j++) {
                b_line = _mm_load_ps(&matrix2[j * size + i]); //_mm_loadu_ps is the non-aligned version that only get a penalty with unaligned memory
                a_line = _mm_set1_ps(matrix1[j + k]);
                r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
            }
            _mm_store_ps(&matrix_out[i + k], r_line);
        }
    }
}

void matrix_mult_sse(int size, double *matrix1, double *matrix2, double *matrix_out) {
    __m128d a_line, b_line, r_line;
    int i, j, k;
# pragma omp parallel               \
    shared(size, matrix1, matrix2, matrix_out)  \
    private(i,j,k, a_line, b_line, r_line)
# pragma omp for
    for (k = 0; k < size * size; k += size) {
# pragma omp for
        for (i = 0; i < size; i += 2) {
            j = 0;
            b_line = _mm_load_pd(&matrix2[i]); //_mm_loadu_ps is the non-aligned version that only get a penalty with unaligned memory
            a_line = _mm_set1_pd(matrix1[j + k]);
            r_line = _mm_mul_pd(a_line, b_line);
            for (j = 1; j < size; j++) {
                b_line = _mm_load_pd(&matrix2[j * size + i]); //_mm_loadu_ps is the non-aligned version that only get a penalty with unaligned memory
                a_line = _mm_set1_pd(matrix1[j + k]);
                r_line = _mm_add_pd(_mm_mul_pd(a_line, b_line), r_line);
            }
            _mm_store_pd(&matrix_out[i + k], r_line);
        }
    }
}