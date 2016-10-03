#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <variant.h>
#include <interactive_tools.h>
#include <user_float.h>
#include <sequential_functions.h>

using namespace std;

void matrix_mult_pl(int size, user_float_t *matrix1_in, user_float_t *matrix2_in, user_float_t *matrix_out) {
# pragma omp parallel				\
	shared(size, matrix1_in, matrix2_in, matrix_out)
# pragma omp for
	for (int cols = 0; cols < size; cols++) {
		for (int rowsOut = 0; rowsOut < size; rowsOut++) {
			matrix_out[cols + rowsOut*size] = 0.0;
			for (int j = 0, rowsIn = 0; rowsIn < size; j++, rowsIn++) {
				matrix_out[cols + rowsOut*size] += matrix1_in[j + rowsOut*size] * matrix2_in[rowsIn*size + cols];
			}
		}
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
		TCLAP::CmdLine cmd("OpenMP Matrix x Matrix Multiplication", ' ', "0.9");

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects,
		// such as "-n Bishop".
		TCLAP::ValueArg<unsigned int> threadsArg("t", "threads", "Number of threads.", true, 2, "threads");
		TCLAP::ValueArg<unsigned int> datasizeArg("s", "data_size", "Data size.", true, 2, "data size");
		TCLAP::ValueArg<unsigned int> iterationsArg("n", "iterations", "The number of iterations.", false, 1, "iterations");
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

		user_float_t *matrix1 = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *matrix2 = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_sq = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		user_float_t *result_pl = (user_float_t *)malloc(sizeof(user_float_t)*size*size);
		matrix_gen(size, matrix1);
		matrix_gen(size, matrix2);

		double time_sq = 0;
		double time_pl = 0;

		omp_set_num_threads(threads);
		time_sq = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			mm_mult_sq(size, matrix1, matrix2, result_sq);
		}
		time_sq = omp_get_wtime() - time_sq;

		time_pl = omp_get_wtime();
		for (unsigned iteration = 0; iteration < iterations; iteration++) {
			matrix_mult_pl(size, matrix1, matrix2, result_pl);
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

		if (debug) {
			cout << "matrix1: " << endl;
			printMatrix(matrix1, size, size);
			cout << "matrix2: " << endl;
			printMatrix(matrix2, size, size);
			cout << "result_sq: " << endl;
			printMatrix(result_sq, size, size);
			cout << "result_pl: " << endl;
			printMatrix(result_pl, size, size);

		}

		//check
		bool checkResult = verifyMatrixResult(result_sq, result_pl, size, debug);

		free(matrix1);
		free(matrix2);
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

