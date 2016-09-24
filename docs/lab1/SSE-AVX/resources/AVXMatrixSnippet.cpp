void matrix_mult_avx(int size, float *matrix1, float *matrix2, float *matrix_out) {
    __m256 a_line, b_line, r_line;
    int i, j, k;
# pragma omp parallel               \
    shared(size, matrix1, matrix2, matrix_out)  \
    private(i,j,k, a_line, b_line, r_line)
# pragma omp for
    for (k = 0; k < size * size; k += size) {
# pragma omp for
        for (i = 0; i < size; i += 8) {
            j = 0;
            b_line = _mm256_load_ps(&matrix2[i]); //_mm256_load_ps is the non-aligned version that only get a penalty with unaligned memory
            a_line = _mm256_set1_ps(matrix1[j + k]);
            r_line = _mm256_mul_ps(a_line, b_line);
            for (j = 1; j < size; j++) {
                b_line = _mm256_load_ps(&matrix2[j * size + i]); //_mm256_load_ps is the non-aligned version that only get a penalty with unaligned memory
                a_line = _mm256_set1_ps(matrix1[j + k]);
                r_line = _mm256_fmadd_ps(a_line, b_line, r_line); // fancy FMA intrinsic, only Haswell and later, FAST though.
                //r_line = _mm256_add_ps(_mm256_mul_ps(a_line, b_line), r_line);
            }
            _mm256_store_ps(&matrix_out[i + k], r_line);
        }
    }
}

void matrix_mult_avx(int size, double *matrix1, double *matrix2, double *matrix_out) {
    __m256d a_line, b_line, r_line;
    int i, j, k;
# pragma omp parallel               \
    shared(size, matrix1, matrix2, matrix_out)  \
    private(i,j,k, a_line, b_line, r_line)
# pragma omp for
    for (k = 0; k < size * size; k += size) {
# pragma omp for
        for (i = 0; i < size; i += 4) {
            j = 0;
            b_line = _mm256_load_pd(&matrix2[i]); //_mm256_load_ps is the non-aligned version that only get a penalty with unaligned memory
            a_line = _mm256_set1_pd(matrix1[j + k]);
            r_line = _mm256_mul_pd(a_line, b_line);
            for (j = 1; j < size; j++) {
                b_line = _mm256_load_pd(&matrix2[j * size + i]); //_mm256_load_ps is the non-aligned version that only get a penalty with unaligned memory
                a_line = _mm256_set1_pd(matrix1[j + k]);
                r_line = _mm256_fmadd_pd(a_line, b_line, r_line); // fancy FMA intrinsic, only Haswell and later, FAST though.
                //r_line = _mm256_add_pd(_mm256_mul_pd(a_line, b_line), r_line);
            }
            _mm256_store_pd(&matrix_out[i + k], r_line);
        }
    }
}