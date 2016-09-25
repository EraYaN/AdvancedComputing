#pragma once
#include "user_float.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <algorithm>

void mv_mult_sq(int size, user_float_t *vector_in, user_float_t *matrix_in, user_float_t *vector_out);
void mm_mult_sq(int size, user_float_t *matrix1_in, user_float_t *matrix2_in, user_float_t *matrix_out);

void matrix_gen(int size, user_float_t *matrix);
void vector_gen(int size, user_float_t *vector);
void matrix_gen(int paddedSize, int size, user_float_t *matrix);
void vector_gen(int size, int sizeReal, user_float_t *vector);