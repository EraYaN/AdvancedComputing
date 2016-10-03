#pragma once
#include <iostream>
#include <iomanip>
#include "user_types.h"

void wait_for_input();

bool verifyVectorResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug = true);

bool verifyMatrixResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug = true);

void printArray(user_float_t arr[], int size, bool all = false);

void printMatrix(user_float_t arr[], int n, int m, bool all = false);