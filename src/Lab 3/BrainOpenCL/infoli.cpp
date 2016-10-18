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
#include "init.h"

using namespace std;

int main(int argc, char *argv[]){

    char *outFileName = "InferiorOlive_Output.txt";
    cl_uint i, j, k , p, q;
    int simSteps = 0;
    int simTime = 0;
    int inputFromFile = 0;
    int initSteps;
    cl_float_t *cellStatePtr;
    cl_float_t *cellCompParamsPtr;
    cl_float_t iApp; 

    int seedvar;
    char temp[100];//warning: this buffer may overflow
    perftime_t t0, t1, tNeighbourStart, tNeighbourEnd, tComputeStart, tComputeEnd, tReadStart, tReadEnd, tWriteStateStart, tWriteStateEnd, tWriteCompStart, tWriteCompEnd, tInitStart, tInitEnd, tLoopStart, tLoopEnd, tWriteFileStart, tWriteFileEnd;
	double tNeighbour, tCompute, tUpdate, tRead, tWriteFile, tInit, tLoop, usecs;
    tNeighbour = tCompute = tUpdate = tRead = tWriteFile = tInit = tLoop = 0;

    cl_event writeDone, neighbourDone, computeDone, readDone;
    cl_int status = 0;

    t0 = now();
    if(EXTRA_TIMING){
        tInitStart = now();    
    }

    simTime = SIMTIME; // in miliseconds
    simSteps = ceil(simTime/DELTA);

    DEBUG_PRINT(("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2));

    //Open output file
	ofstream pOutFile(outFileName);

	if (!pOutFile) {
		cerr << "Error: Couldn't create " << outFileName << endl;
		exit(EXIT_FAILURE);
	}
	pOutFile << "#simSteps Time(ms) Input(Iapp) Output(V_axon)" << endl; 

    //Malloc for the array of cellStates and cellCompParams
    mallocCells(&cellCompParamsPtr, &cellStatePtr);

    //Write initial state values
    InitState(cellStatePtr);

    //Initialize g_CaL
    init_g_CaL(cellStatePtr);


    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------


    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //-----------------------------------------------------


    //-----------------------------------------------------
    // STEP 3: Create a context
    //-----------------------------------------------------


    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //-----------------------------------------------------


    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //-----------------------------------------------------
 

    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------

    //-----------------------------------------------------
    // STEP 7: Create and compile the program
    //-----------------------------------------------------
    /**
    ** Please check if your kernel files can be opened, 
    ** especially when running on the server
    **/ 
    char *computeFileName, *neighbourFileName;
    neighbourFileName = "neighbour_kernel.cl";
    computeFileName = "compute_kernel.cl";

	ifstream neighbourFile(neighbourFileName);
    if(!neighbourFile){
        cerr << "cannot open neighbour file" << endl;
		cerr << "current path:" << neighbourFileName << endl;
        exit(EXIT_FAILURE);
    }

	ifstream computeFile(computeFileName);
    if(!computeFile){
		cerr << "cannot open neighbour file" << endl;
		cerr << "current path:" << neighbourFileName << endl;
        exit(EXIT_FAILURE);
    }

    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //----------------------------------------------------


    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //-----------------------------------------------------


    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //-----------------------------------------------------


    for(i=0;i<simSteps;i++){
        //Compute one sim step for all cells
        if(i>20000-1 && i<20500-1){ iApp = 6;} // start @ 1 because skipping initial values
        else{ iApp = 0;}
		pOutFile << string_format("%d %.2f %.1f ", i + 1, i*0.05, iApp) << endl;        

        if(EXTRA_TIMING){
            tNeighbourStart = now();
        }

        //-----------------------------------------------------
        // STEP 11.1: Run neighbour kernel
        //-----------------------------------------------------


        if(EXTRA_TIMING){
            status = clWaitForEvents(1, &neighbourDone);
            tNeighbourEnd = now();
            tNeighbour += diffToNanoseconds(tNeighbourStart, tNeighbourEnd);
            tComputeStart = now(); 
        }

        //-----------------------------------------------------
        // STEP 11.2: Run compute kernel
        //-----------------------------------------------------

        if(EXTRA_TIMING){
            status = clWaitForEvents(1, &computeDone);
            tComputeEnd = now();
            tCompute += diffToNanoseconds(tComputeStart, tComputeEnd);
            tReadStart = now();
        }

        if(status != CL_SUCCESS){
            cerr << "error in loop, compute" << endl;
            exit(EXIT_FAILURE);
        }

        
        // transfer data from device to CPU
        if(WRITE_OUTPUT){
            //-----------------------------------------------------
            // STEP 11.3: Read output data from device
            //-----------------------------------------------------
        }

        if(EXTRA_TIMING){
            tReadEnd = now();
            tRead += diffToNanoseconds(tReadStart, tReadEnd);
        }

        // write output to file
        if(WRITE_OUTPUT){
            if(EXTRA_TIMING){
                tWriteFileStart = now();
            }
            for(j = 0; j < IO_NETWORK_DIM1; j++){
                for(k = 0; k < IO_NETWORK_DIM2; k++){
					pOutFile << setprecision(8) << cellStatePtr[((((i % 2) ^ 1)*IO_NETWORK_SIZE + (j + k*IO_NETWORK_DIM1))* STATE_SIZE) + AXON_V] << " ";
                }                
            }
			pOutFile << endl;
            if(EXTRA_TIMING){
                tWriteFileEnd = now();
                tWriteFile += diffToNanoseconds(tWriteFileStart, tWriteFileEnd);
            }
        }
    }
    if(EXTRA_TIMING){
        tLoopEnd = now();
    }
    
    t1 = now();
    usecs = diffToNanoseconds(t0,t1)/1e3;// / 1000000;
    DEBUG_PRINT(("%d ms of brain time in %d simulation steps\n", simTime, simSteps));
    DEBUG_PRINT((" %lld usecs real time \n", usecs));

    if(EXTRA_TIMING){
        tInit = diffToNanoseconds(tInitStart, tInitEnd);
        tLoop = diffToNanoseconds(tLoopStart, tLoopEnd);
        
        DEBUG_PRINT(("\n"));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tInit: \t\t %.1f s\n", tInit/1e9));
        DEBUG_PRINT(("tLoop: \t\t %.1f s\n", tLoop / 1e9));
        DEBUG_PRINT(("\ttNeighbour: \t %.1f s\n", tNeighbour / 1e9));
        DEBUG_PRINT(("\ttCompute: \t %.1f s\n", tCompute / 1e9));
        DEBUG_PRINT(("\ttRead: \t\t %.1f s\n", tRead / 1e9));
        DEBUG_PRINT(("\ttWriteFile: \t %.1f s\n", tWriteFile / 1e9));
        DEBUG_PRINT(("\t----------- + \n"));
        DEBUG_PRINT(("\ttSumLoop: \t %.1f s\n", (tWriteFile + tCompute + tNeighbour + tRead) / 1e9));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tSum: \t %.1f s\n", (tInit + tLoop) / 1e9));
    }
    
    //-----------------------------------------------------
    // STEP 12: Release OpenCL resources
    //----------------------------------------------------- 


    //Free up memory and close files
    free(cellStatePtr);
    free(cellCompParamsPtr);
    pOutFile.close();

    return EXIT_SUCCESS;
}

