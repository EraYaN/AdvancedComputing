#include <iostream>
#include <cassert>
#include <windows.h>

using namespace std;

struct StructName {
	double val;
	StructName() : val(0) {}
	StructName(double initValue) : val(initValue) {}
};

void makeStruct(StructName *s) {
	StructName s2 = StructName(25);
	*s = s2;
}

int main() {
	StructName test = StructName(0);
	StructName test2 = StructName(25);

	StructName *test3;

	Sleep(1000);
	//does not leak
	for (int i = 0;i < 10000;i++) {
		makeStruct(&test);
	}

	Sleep(1000);
	//leaks
	for (int i = 0;i < 10000;i++) {
		test3 = new StructName(45);
	}

	Sleep(1000);

	return 0;

}