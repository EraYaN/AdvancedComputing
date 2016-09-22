#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include <variant.h>
#include <interactive_tools.h>

using namespace std;
// generate two input matrices
void matrix_matrix_gen(int size, double *matrix1, double *matrix2) {
	int i;
	for (i = 0; i < size*size; i++)
		matrix1[i] = ((double)rand()) / 5307.0;
	for (i = 0; i < size*size; i++)
		matrix2[i] = ((double)rand()) / 5307.0;
}

// matrix matrix multiplication
void matrix_mult_sq(int size, double *matrix1_in,
	double *matrix2_in, double *matrix_out) {
	int rowsOut, rowsIn, cols;
	int j;
	for (cols = 0; cols<size; cols++) {
		for (rowsOut = 0; rowsOut<size; rowsOut++) {
			matrix_out[cols + rowsOut*size] = 0.0;
			for (j = 0, rowsIn = 0; rowsIn<size; j++, rowsIn++) {
				matrix_out[cols + rowsOut*size] += matrix1_in[j + rowsOut*size] * matrix2_in[rowsIn*size + cols];
			}
		}
	}
}

void matrix_mult_pl(int size, double *matrix1_in,
	double *matrix2_in, double *matrix_out) {
	int rowsOut, rowsIn, cols;
	int j;
# pragma omp parallel				\
	shared(size, matrix1_in, matrix2_in, matrix_out)	\
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

		double *matrix1 = (double *)malloc(sizeof(double)*size*size);
		double *matrix2 = (double *)malloc(sizeof(double)*size*size);
		double *result_sq = (double *)malloc(sizeof(double)*size*size);
		double *result_pl = (double *)malloc(sizeof(double)*size*size);
		matrix_matrix_gen(size, matrix1, matrix2);

		double time_sq = 0;
		double time_pl = 0;

		omp_set_num_threads(threads);
		time_sq = omp_get_wtime();
		matrix_mult_sq(size, matrix1, matrix2, result_sq);
		time_sq = omp_get_wtime() - time_sq;

		time_pl = omp_get_wtime();
		matrix_mult_pl(size, matrix1, matrix2, result_pl);
		time_pl = omp_get_wtime() - time_pl;

		if (debug) {
			printf("DAT:%d\n", size);
			printf("THD:%d\n", omp_get_max_threads());
			printf("PRC:%d\n", omp_get_num_procs());
		}
		printf("SEQ:%.14f\n", time_sq);
		printf("VAR:%.14f\n", time_pl);

		//check
		for (unsigned int i = 0; i < size; i++)
			if (result_sq[i] != result_pl[i]) {
				if (debug) {
					cout << "Wrong value \"" << result_sq[i] << "\" and \"" << result_pl[i] << "\" at position " << i << "." << endl;
				}
				if (interactive) {
					wait_for_input();
				}
				return 3;
			}

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
		return 0;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
		return -1;
	}
}

