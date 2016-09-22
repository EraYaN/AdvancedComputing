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
void matrix_vector_gen(int size, float *matrix, float *vector) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = i*1.2f + 1;//((float)rand())/65535.0f;
	for (i = 0; i < size*size; i++)
		matrix[i] = i*1.3f + 1;//((float)rand())/5307.0f;
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, float *vector_in,
	float *matrix_in, float *vector_out) {
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


int main(int argc, char *argv[]) {

	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {
		// Define the command line object, and insert a message
		// that describes the program. The "Command description message"
		// is printed last in the help text. The second argument is the
		// delimiter (usually space) and the last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects
		// that it contains.
		TCLAP::CmdLine cmd("OpenMP Matrix x Vector Multiplication", ' ', "0.9");

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<unsigned int> threadsArg("t", "threads", "Number of threads.", true, 2, "unsigned int");
		TCLAP::ValueArg<unsigned int> datasizeArg("s", "data_size", "Data size.", true, 2, "unsigned int");
		TCLAP::ValuesConstraint<int> variantConstraint(variants);
		TCLAP::ValueArg<int> variantArg("v", "variant", "Variant ID to run.", false, (int)base, &variantConstraint, false);
		TCLAP::SwitchArg debugArg("d", "debug", "Enable debug mode, verbose output.", false);
		TCLAP::SwitchArg interactiveArg("i", "interactive", "Enable interactive mode.", false);

		// Add the argument nameArg to the CmdLine object. The CmdLine object
		// uses this Arg to parse the command line.
		cmd.add(threadsArg);
		cmd.add(datasizeArg);
		cmd.add(variantArg);
		cmd.add(debugArg);
		cmd.add(interactiveArg);

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each arg.
		unsigned size = datasizeArg.getValue();
		unsigned threads = threadsArg.getValue();
		Variant variant = (Variant)variantArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		if (size % 4 != 0) {
			printf("This version implements for ""size = 4*n"" only\n");
			return 2;
		}

		//if (size % 4 != 0) {
		//	// round to nearest multiple of 4 to enable aligned_malloc
		//	size += 4 - (size % 4);

		//	//printf("This version implements for ""size = 4*n"" only\n");
		//	//return 2;
		//}

		float *vector = (float *)aligned_malloc(sizeof(float) * 4, sizeof(float)*size);//(float *)malloc(sizeof(float)*size);
		if (vector == NULL) {
			printf("can't allocate the required memory for vector\n");
			return 3;
		}

		float *matrix = (float *)aligned_malloc(sizeof(float) * 4, sizeof(float)*size*size);
		if (matrix == NULL) {
			printf("can't allocate the required memory for matrix\n");
			aligned_free(vector);
			return 4;
		}

		float *result_sq = (float *)aligned_malloc(sizeof(float) * 4, sizeof(float)*size);
		if (result_sq == NULL) {
			printf("can't allocate the required memory for result_sq\n");
			aligned_free(vector);
			aligned_free(matrix);
			return 5;
		}

		float *result_pl = (float *)aligned_malloc(sizeof(float) * 4, sizeof(float)*size);
		if (result_pl == NULL) {
			printf("can't allocate the required memory for result_pl\n");
			aligned_free(vector);
			aligned_free(matrix);
			aligned_free(result_sq);
			return 6;
		}

		matrix_vector_gen(size, matrix, vector);

		double time_sq;
		double time_sse;

		omp_set_num_threads(threads);

		time_sq = omp_get_wtime();
		matrix_mult_sq(size, vector, matrix, result_sq);
		time_sq = omp_get_wtime() - time_sq;

		time_sse = omp_get_wtime();
		matrix_mult_sse(size, vector, matrix, result_pl);
		time_sse = omp_get_wtime() - time_sse;

		if (debug) {
			printf("DAT:%d\n", size);
			printf("THD:%d\n", omp_get_max_threads());
			printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_sse);

		//check
		/*int i;
		for(i=0; i<size; i++)
		  if((int)result_sq[i] != (int)result_pl[i]){
			printf("wrong at position %d\n", i);
			aligned_free(vector);
			aligned_free(matrix);
			aligned_free(result_sq);
			aligned_free(result_pl);
			return 7;
		  }*/

		aligned_free(vector);
		aligned_free(matrix);
		aligned_free(result_sq);
		aligned_free(result_pl);
		return 0;
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return -1;
	}
}
