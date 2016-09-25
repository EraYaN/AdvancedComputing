#pragma once
#include <iostream>
#include <iomanip>
#include "user_float.h"

#define EXIT_BADARGUMENT 2
#define EXIT_WRONGVALUE 3
#define EXIT_OPENCLERROR 4
#define EXIT_MEMORYERROR 5

void wait_for_input();

bool verifyVectorResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug = true);

bool verifyMatrixResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug = true);

void printArray(user_float_t arr[], int size, bool all = false);

void printMatrix(user_float_t arr[], int n, int m, bool all = false);

