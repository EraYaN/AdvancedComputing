#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <variant.h>
#include <interactive_tools.h>
#include <user_float.h>

using namespace std;
/*****************************************************
the following function generates a "size"-element vector
and a "size x size" matrix
 ****************************************************/
void matrix_vector_gen(int size, user_float_t *matrix, user_float_t *vector) {
	int i;
	for (i = 0; i < size; i++)
		vector[i] = ((user_float_t)rand()) / 65535.0;
	for (i = 0; i < size*size; i++)
		matrix[i] = ((user_float_t)rand()) / 5307.0;
}

/****************************************************
the following function calculate the below equation
   vector_out = vector_in x matrix_in
 ***************************************************/
void matrix_mult_sq(int size, user_float_t *vector_in, user_float_t *matrix_in, user_float_t *vector_out) {
	int rows, cols;
	int j;
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
	}
}

void matrix_mult_pl(int size, user_float_t *vector_in, user_float_t *matrix_in, user_float_t *vector_out) {
	int rows, cols;
	int j;
# pragma omp parallel				\
  shared(size, vector_in, matrix_in, vector_out)	\
  private(rows, cols, j)
# pragma omp for
	for (cols = 0; cols < size; cols++) {
		vector_out[cols] = 0.0;
		for (j = 0, rows = 0; rows < size; j++, rows++)
			vector_out[cols] += vector_in[j] * matrix_in[rows*size + cols];
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
		unsigned threads = threadsArg.getValue();
		unsigned iterations = iterationsArg.getValue();
		Variant variant = (Variant)variantArg.getValue();
		bool debug = debugArg.getValue();
		bool interactive = interactiveArg.getValue();

		user_float_t *vector = (user_float_t *)malloc(sizeof(user_float_t)*size);
		user_float_t *matrix = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_sq = (user_float_t *)malloc(sizeof(user_float_t)*size);
		user_float_t *result_pl = (user_float_t *)malloc(sizeof(user_float_t)*size);
		matrix_vector_gen(size, matrix, vector);

		double time_sq = 0;
		double time_pl = 0;

		omp_set_num_threads(threads);
		time_sq = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_sq(size, vector, matrix, result_sq);
		}
		time_sq = omp_get_wtime() - time_sq;

		time_pl = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_pl(size, vector, matrix, result_pl);
		}
		time_pl = omp_get_wtime() - time_pl;

		if (debug) {
			printf("ITR:%d\n", iterations);
			printf("DAT:%d\n", size);
			printf("THD:%d\n", omp_get_max_threads());
			printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_pl);

		//check
		bool checkResult = verifyVectorResult(result_sq, result_pl, size, debug);

		free(vector);
		free(matrix);
		free(result_sq);
		free(result_pl);
		if (debug) {
			cout << "Done." << endl;
		}
		if (interactive) {
			wait_for_input();
		}

		if (checkResult)
			return EXIT_SUCCESS;
		else
			return EXIT_WRONGVALUE;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return EXIT_BADARGUMENT;
	}
}
