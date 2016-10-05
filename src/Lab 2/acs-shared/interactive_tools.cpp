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
