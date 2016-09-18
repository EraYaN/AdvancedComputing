#include "Tools.h"

int ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	{ 0x30, 192 }, // Fermi/Kepler Generation (SM 3.0) GK10x class
	{ 0x35, 192 }, // Kepler Generation (SM 3.0) GK11x class
	{ 0x50, 128 }, // Kepler Generation (SM 5.0) GK20x classes
	{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM10x & GM20x class
	{ 0x53, 128 }, // Pascal Generation (SM 5.3) GP10x class
	{ -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}