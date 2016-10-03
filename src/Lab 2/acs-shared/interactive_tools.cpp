#include "interactive_tools.h"
using namespace std;

#define FORMAT_SEPARATOR ' '
#define FORMAT_NUMWIDTH 12
#define FORMAT_PRECISION 6
#define FORMAT_LIMIT 10

void wait_for_input() {
	cout << "Press enter to continue." << endl;
	cin.get();
}

void printArray(user_float_t arr[], int size, bool all) {
	int limit = size;
	if (!all&&size > FORMAT_LIMIT)
		limit = FORMAT_LIMIT;

	for (int i = 0; i < limit; i++) {
		cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[i];
	}
	if (!all&&size > FORMAT_LIMIT)
		cout << "[..]";
	cout << endl;
}

void printMatrix(user_float_t arr[], int rows, int cols, bool all){
	int limit_cols = cols;
	if (!all&&cols > FORMAT_LIMIT)
		limit_cols = FORMAT_LIMIT;
	int limit_rows = rows;
	if (!all&&rows > FORMAT_LIMIT)
		limit_rows = FORMAT_LIMIT;

	for (int col = 0; col < limit_cols; col++) {
		for (int row = 0; row < limit_rows; row++) {
			cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[row*rows + col];
		}
		if (!all&&rows > FORMAT_LIMIT)
			cout << "[..]";
		cout << endl;
	}
	if (!all&&cols > FORMAT_LIMIT)
		cout << "[.., ..]" << endl;
}

bool verifyVectorResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug) {
	for (unsigned i = 0; i < size; i++) {
		if ((int)round(result1[i]) != (int)round(result2[i])) {
			if (debug) {
				cout << "Wrong value \"" << (int)round(result1[i]) << "\" and \"" << (int)round(result2[i]) << "\" at position " << i << "." << endl;
			}
			return false;
		}
	}
	return true;
}

bool verifyMatrixResult(user_float_t result1[], user_float_t result2[], unsigned size, bool debug) {
	unsigned i;
	for (i = 0; i < size * size; i++) {
		if ((int)round(result1[i]) != (int)round(result2[i])) {
			if (debug) {
				cout << "Wrong value \"" << (int)round(result1[i]) << "\" and \"" << (int)round(result2[i]) << "\" at position (" << i % size << ',' << (int)floor((double)i / size) << ")." << endl;
			}
			return false;
		}
	}
	return true;
}