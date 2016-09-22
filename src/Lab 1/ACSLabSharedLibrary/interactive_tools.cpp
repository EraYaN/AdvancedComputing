#include "interactive_tools.h"
using namespace std;

#define FORMAT_SEPARATOR ' '
#define FORMAT_NUMWIDTH 12
#define FORMAT_PRECISION 3

void wait_for_input() {
	cout << "Press enter to continue." << endl;
	cin.get();
}

void printArray(double arr[], int size) {
	for (int i = 0; i < size; i++) {
		cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[i];
	}
	cout << endl;
}

void printArray(float arr[], int size) {
	for (int i = 0; i < size; i++) {
		cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[i];
	}
	cout << endl;
}

void printMatrix(double arr[], int rows, int cols){
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[row*rows + col];
		}
		cout << endl;
	}
}

void printMatrix(float arr[], int rows, int cols) {
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			cout << left << setprecision(FORMAT_PRECISION) << setw(FORMAT_NUMWIDTH) << setfill(FORMAT_SEPARATOR) << arr[row*rows + col];
		}
		cout << endl;
	}
}