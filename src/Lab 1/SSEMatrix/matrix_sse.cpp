#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <xmmintrin.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <variant.h>
#include <interactive_tools.h>
#include <user_float.h>

using namespace std;

#ifdef __unix__
#define aligned_malloc(alignment, size)    memalign(alignment,size)
#define aligned_free(ptr)    free(ptr)
#elif defined(_WIN32) || defined(WIN32)     /* _Win32 is usually defined by compilers targeting 32 or   64 bit Windows systems */
#define aligned_malloc(alignment, size)    _aligned_malloc(size,alignment)
#define aligned_free(ptr)    _aligned_free(ptr)
#endif

/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
****************************************************/
void matrix_vector_gen(int size, int sizeReal, user_float_t *matrix, user_float_t *vector) {
	int i;
	for (i = 0; i < size; i++) {
		if (i < sizeReal) {
			vector[i] = i*1.2f + 1;//((float)rand())/65535.0f;
		} else {
			vector[i] = 0.0;
		}
	}

	for (i = 0; i < size*size; i++) {
		if (i % size < sizeReal && i < size * sizeReal) {
			matrix[i] = i*1.3f + 1;//((float)rand())/5307.0f;
		} else {
			matrix[i] = 0.0;
		}
	}
}

/****************************************************
the following function calculate the below equation
vector_out = vector_in x matrix_in
***************************************************/
void matrix_mult_sq(int size, user_float_t *vector_in,
	user_float_t *matrix_in, user_float_t *vector_out) {
	int rows, cols;
	int j;
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

void matrix_mult_sse(int size, float *vector_in,
	float *matrix_in, float *vector_out) {
	__m128 a_line, b_line, r_line;
	int i, j;
	for (i = 0; i < size; i += 4) {
		j = 0;
		b_line = _mm_load_ps(&matrix_in[i]); // b_line = vec4(matrix[i][0])
		a_line = _mm_set1_ps(vector_in[j]);      // a_line = vec4(vector_in[0])
		r_line = _mm_mul_ps(a_line, b_line); // r_line = a_line * b_line
		for (j = 1; j < size; j++) {
			b_line = _mm_load_ps(&matrix_in[j*size + i]); // a_line = vec4(column(a, j))
			a_line = _mm_set1_ps(vector_in[j]);  // b_line = vec4(b[i][j])
												 // r_line += a_line * b_line
			r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
		}
		_mm_store_ps(&vector_out[i], r_line);     // r[i] = r_line
	}
}

void matrix_mult_sse(int size, double *vector_in,
	double *matrix_in, double *vector_out) {
	__m128d a_line, b_line, r_line;
	int i, j;
	for (i = 0; i < size; i += 2) {
		j = 0;
		b_line = _mm_load_pd(&matrix_in[i]); // b_line = vec4(matrix[i][0])
		a_line = _mm_set1_pd(vector_in[j]);      // a_line = vec4(vector_in[0])
		r_line = _mm_mul_pd(a_line, b_line); // r_line = a_line * b_line
		for (j = 1; j < size; j++) {
			b_line = _mm_load_pd(&matrix_in[j*size + i]); // a_line = vec4(column(a, j))
			a_line = _mm_set1_pd(vector_in[j]);  // b_line = vec4(b[i][j])
												 // r_line += a_line * b_line
			r_line = _mm_add_pd(_mm_mul_pd(a_line, b_line), r_line);
		}
		_mm_store_pd(&vector_out[i], r_line);     // r[i] = r_line
	}
}


int main(int argc, char *argv[]) {
	const int ALIGNMENT_SIZE = 4;
	const int ALIGNMENT_BYTES = sizeof(user_float_t) * ALIGNMENT_SIZE;
	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {
		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.

#ifdef USE_DOUBLES
		TCLAP::CmdLine cmd("SSE Matrix x Vector Multiplication (Double Precision)", ' ', "0.9");
#else
		TCLAP::CmdLine cmd("SSE Matrix x Vector Multiplication (Single Precision)", ' ', "0.9");
#endif

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<unsigned int> threadsArg("t", "threads", "Number of threads.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> datasizeArg("s", "data_size", "Data size.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> iterationsArg("n", "iterations", "The number of iterations.", false, 1, "unsigned int");
		TCLAP::ValuesConstraint<int> variantConstraint(variants);
		TCLAP::ValueArg<int> variantArg("v", "variant", "Variant ID to run.", false, (int)base, &variantConstraint, false);
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(threadsArg);
		cmd.add(datasizeArg);
		cmd.add(iterationsArg);
		cmd.add(variantArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		unsigned size = datasizeArg.getValue();
		unsigned paddedSize = size;
		unsigned threads = threadsArg.getValue();
		unsigned iterations = iterationsArg.getValue();
		Variant variant = (Variant)variantArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		if (variant == base) {
			if (size % 4 != 0) {
				printf("This version implements for ""size = 4*n"" only\n");
				if (interactive) wait_for_input();
				return 2;
			}
		} else if (variant == arbitrarysize) {
			if (size % ALIGNMENT_SIZE != 0) {
				paddedSize = size + (ALIGNMENT_SIZE - size % ALIGNMENT_SIZE);
			}
		}
		user_float_t *vector;
		user_float_t *matrix;
		user_float_t *result_sq;
		user_float_t *result_pl;


		size_t vectorSize = sizeof(user_float_t)*paddedSize;
		size_t matrixSize = sizeof(user_float_t)*paddedSize*paddedSize;
		vector = (user_float_t *)aligned_malloc(ALIGNMENT_BYTES, vectorSize);//(float *)malloc(sizeof(float)*size);
		matrix = (user_float_t *)aligned_malloc(ALIGNMENT_BYTES, matrixSize);
		result_sq = (user_float_t *)aligned_malloc(ALIGNMENT_BYTES, vectorSize);
		result_pl = (user_float_t *)aligned_malloc(ALIGNMENT_BYTES, vectorSize);

		if (vector == NULL) {
			printf("can't allocate the required memory for vector\n");
			return 3;
		}
		if (matrix == NULL) {
			printf("can't allocate the required memory for matrix\n");
			aligned_free(vector);
			return 4;
		}
		if (result_sq == NULL) {
			printf("can't allocate the required memory for result_sq\n");
			aligned_free(vector);
			aligned_free(matrix);
			return 5;
		}
		if (result_pl == NULL) {
			printf("can't allocate the required memory for result_pl\n");
			aligned_free(vector);
			aligned_free(matrix);
			aligned_free(result_sq);
			return 6;
		}
		matrix_vector_gen(paddedSize, size, matrix, vector);


		double time_sq;
		double time_sse;

		omp_set_num_threads(threads);

		time_sq = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_sq(paddedSize, vector, matrix, result_sq);
		}
		time_sq = omp_get_wtime() - time_sq;

		time_sse = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_sse(paddedSize, vector, matrix, result_pl);
		}
		time_sse = omp_get_wtime() - time_sse;

		if (debug) {
			printf("ITR:%d\n", iterations);
			printf("DAT:%d\n", size);
			printf("THD:%d\n", omp_get_max_threads());
			printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_sse);


		if (debug) {
			cout << "vector: " << endl;
			printArray(vector, paddedSize);
			cout << "matrix: " << endl;
			printMatrix(matrix, paddedSize, paddedSize);
			cout << "result_sq: " << endl;
			printArray(result_sq, paddedSize);
			cout << "result_pl: " << endl;
			printArray(result_pl, paddedSize);

		}

		//check
		for (unsigned i = 0; i < size; i++)
			if ((int)result_sq[i] != (int)result_pl[i]) {
				if (debug) {
					cout << "Wrong value \"" << result_sq[i] << "\" and \"" << result_pl[i] << "\" at position " << i << "." << endl;
				}
				aligned_free(vector);
				aligned_free(matrix);
				aligned_free(result_sq);
				aligned_free(result_pl);
				if (interactive) wait_for_input();
				return 7;
			}

		aligned_free(vector);
		aligned_free(matrix);
		aligned_free(result_sq);
		aligned_free(result_pl);
		if (interactive) wait_for_input();
		return 0;
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return -1;
	}
}
