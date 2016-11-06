/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 *
 * Any use or reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 *
 * Author: Sebastian Isaza
 * Created: 19-01-2012
 * Modified: 07-08-2012
 *
 * Description: Top source file of the Inferior Olive model, originally written
 * in Matlab by Jornt De Gruijl. It contains the implementation of all functions.
 * The main function allocates the necessary memory, initializes the system
 * state and runs the model calculations.
 *
 */
#include "infoli.h"

#define CSV_SEPARATOR ','
#define LINE_MARKER '@'

using namespace std;

int main(int argc, char *argv[]) {

	const char* outFileName = "../InferiorOlive_Output_CPU.txt";
	int i, j, k, p, q;
	int simSteps = 0;
	int simTime = 0;
	int inputFromFile = 0;
	int initSteps;
	cellState ***cellStatePtr;
	cellCompParams **cellCompParamsPtr;
	int seedvar;
	char temp[100];//warning: this buffer may overflow
	user_float_t iApp;
	perftime_t t0, t1, tNeighbourStart, tNeighbourEnd, tComputeStart, tComputeEnd, tInitStart, tInitEnd, tLoopStart, tLoopEnd, tWriteFileStart, tWriteFileEnd;
	double tNeighbour, tCompute, tRead, tWriteFile, tInit, tLoop, usecs;
	tNeighbour = tCompute = tWriteFile = tInit = tLoop = 0;
	//double secs;

	t0 = now();
	if (EXTRA_TIMING) {
		tInitStart = now();
	}
	DEBUG_PRINT(("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2));

	ofstream pOutFile;
	if (WRITE_OUTPUT) {
		//Open output file
		pOutFile.open(outFileName);

		if (!pOutFile) {
			cerr << "Error: Couldn't create " << outFileName << endl;
			exit(EXIT_FAILURE);
		}
		pOutFile << "#simSteps Time(ms) Input(Iapp) Output(V_axon)" << endl;
	}


	//Malloc for the array of cellStates and cellCompParams
	mallocCells(&cellCompParamsPtr, &cellStatePtr);
	//Write initial state values
	InitState(cellStatePtr[0]);

	//Initialize g_CaL
	init_g_CaL(cellStatePtr);
	//Random initialization: put every cell in a different oscillation state
	if (RAND_INIT) {
		random_init(cellCompParamsPtr, cellStatePtr);
	}

	simTime = SIMTIME; // in miliseconds
	simSteps = ceil(simTime / DELTA);

	if (EXTRA_TIMING) {
		tInitEnd = now();
		tLoopStart = now();
	}

	for (i = 0;i < simSteps;i++) {
		//Compute one sim step for all cells
		if (i > 20000 - 1 && i < 20500 - 1) { iApp = 6; } // start @ 1 because skipping initial values
		else { iApp = 0; }
		if (WRITE_OUTPUT) {
			pOutFile << string_format("%d %.2f %.1f ", i + 1, i*0.05, iApp);
		}
		for (j = 0;j < IO_NETWORK_DIM1;j++) {
			for (k = 0;k < IO_NETWORK_DIM2;k++) {
				if (EXTRA_TIMING) {
					tNeighbourStart = now();
				}
				neighbors(cellCompParamsPtr, cellStatePtr, i, j, k);
				if (EXTRA_TIMING) {
					tNeighbourEnd = now();
					tNeighbour += diffToNanoseconds(tNeighbourStart, tNeighbourEnd);
					tComputeStart = now();
				}
				compute(cellCompParamsPtr, cellStatePtr, iApp, i, j, k);
				if (EXTRA_TIMING) {
					tComputeEnd = now();
					tCompute += diffToNanoseconds(tComputeStart, tComputeEnd);
					tWriteFileStart = now();
				}
				if (WRITE_OUTPUT) {
					pOutFile << setprecision(8) << cellStatePtr[(i % 2) ^ 1][j][k].axon.V_axon << " ";
				}
				if (EXTRA_TIMING) {
					tWriteFileEnd = now();
					tWriteFile += diffToNanoseconds(tWriteFileStart, tWriteFileEnd);
				}
			}
		}
		if (EXTRA_TIMING) {
			tWriteFileStart = now();
		}
		if (WRITE_OUTPUT) {
			pOutFile << endl;
		}
		if (EXTRA_TIMING) {
			tWriteFileEnd = now();
			tWriteFile += diffToNanoseconds(tWriteFileStart, tWriteFileEnd);
		}
		//wait_for_input();

		//if (i >= 2) break;
	}
	if (EXTRA_TIMING) {
		tLoopEnd = now();
	}

	t1 = now();
	usecs = diffToNanoseconds(t0, t1) / 1e3;
	DEBUG_PRINT(("%d ms of brain time in %d simulation steps\n", simTime, simSteps));
	DEBUG_PRINT((" %f usecs real time \n", usecs));

	if (EXTRA_TIMING) {
		tInit = diffToNanoseconds(tInitStart, tInitEnd);
		tLoop = diffToNanoseconds(tLoopStart, tLoopEnd);

		DEBUG_PRINT(("\n"));
		DEBUG_PRINT(("----------------------------------\n"));
		DEBUG_PRINT(("tInit: \t\t %.1f s\n", tInit / 1e9));
		DEBUG_PRINT(("tLoop: \t\t %.1f s\n", tLoop / 1e9));
		DEBUG_PRINT(("\ttNeighbour: \t %.1f s\n", tNeighbour / 1e9));
		DEBUG_PRINT(("\ttCompute: \t %.1f s\n", tCompute / 1e9));
		DEBUG_PRINT(("\ttWriteFile: \t %.1f s\n", tWriteFile / 1e9));
		DEBUG_PRINT(("\t----------- + \n"));
		DEBUG_PRINT(("\ttSumLoop: \t %.1f s\n", (tWriteFile + tCompute + tNeighbour) / 1e9));
		DEBUG_PRINT(("----------------------------------\n"));
		DEBUG_PRINT(("tSum: \t %.1f s\n", (tInit + tLoop) / 1e9));
		cout << LINE_MARKER << "CPU" << CSV_SEPARATOR << tInit / 1e9 << CSV_SEPARATOR << tNeighbour / 1e9 << CSV_SEPARATOR << tCompute / 1e9 << CSV_SEPARATOR << tWriteFile / 1e9 << CSV_SEPARATOR << (tInit + tLoop) / 1e9 << endl;
	}

	//Free up memory and close files
	free(cellStatePtr[0]);
	free(cellStatePtr[1]);
	free(cellStatePtr);
	free(cellCompParamsPtr);
	if (WRITE_OUTPUT) {
		pOutFile.close();
	}

	//wait_for_input();

	return EXIT_SUCCESS;
}

/**
Input: cellCompParamsPtr, cellStatePtr, i, j, k
cellCompParamsPtr: Array of struct which stores values of neighbours for each cell.
cellStatePtr: Array with values for each cell.
i: current simulation step
j: current position in dimension 1 of the IO network
k: current position in dimension 2 of the IO network

Retreive the voltage of the dendrite (V_dend) from each neighbour
**/
void neighbors(cellCompParams **cellCompParamsPtr, cellState ***cellStatePtr, int i, int j, int k) {
	int n, p, q;
	n = 0;
	for (p = j - 1;p <= j + 1;p++) {
		for (q = k - 1;q <= k + 1;q++) {
			if (((p != j) || (q != k)) && ((p >= 0) && (q >= 0)) && ((p < IO_NETWORK_DIM1) && (q < IO_NETWORK_DIM2))) {
				cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i % 2][p][q].dend.V_dend;
			} else if (p == j && q == k) {
				;   // do nothing, this is the cell itself
			} else {
				//store same V_dend so that Ic becomes zero by the subtraction
				cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i % 2][j][k].dend.V_dend;
			}
		}
	}
}

/**
Input: cellCompParamsPtr, cellStatePtr, iApp ,i, j, k
cellCompParamsPtr: Array of struct which stores values of neighbours for each cell.
cellStatePtr: Array with values for each cell.
iApp: Extenal input of the dendrite
i: Current simulation step
j: Current position in dimension 1 of the IO network
k: Current position in dimension 2 of the IO network

Retreive the external input of the dedrite
and update the previous and new state of the current cell.
Then Compute the new variables of the current cell with ComputeOneCell.
**/
void compute(cellCompParams **cellCompParamsPtr, cellState ***cellStatePtr, int iApp, int i, int j, int k) {
	cellCompParamsPtr[j][k].iAppIn = iApp;
	cellCompParamsPtr[j][k].prevCellState = &cellStatePtr[i % 2][j][k];
	cellCompParamsPtr[j][k].nextCellState = &cellStatePtr[(i % 2) ^ 1][j][k];
	//Compute one Cell...
	ComputeOneCell(&cellCompParamsPtr[j][k]);
}

