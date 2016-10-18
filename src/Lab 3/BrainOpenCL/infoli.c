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
#include "ioFile.h"

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp ()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int main(int argc, char *argv[]){

    char *outFileName = "InferiorOlive_Output.txt";
    FILE *pOutFile;
    cl_uint i, j, k , p, q;
    int simSteps = 0;
    int simTime = 0;
    int inputFromFile = 0;
    int initSteps;
    cl_mod_prec *cellStatePtr;
    cl_mod_prec *cellCompParamsPtr;
    cl_mod_prec iApp; 

    int seedvar;
    char temp[100];//warning: this buffer may overflow
    timestamp_t t0, t1, usecs, tNeighbourStart, tNeighbourEnd, tComputeStart, tComputeEnd, tReadStart, tReadEnd, tWriteStateStart, tWriteStateEnd, tWriteCompStart, tWriteCompEnd, tInitStart, tInitEnd, tLoopStart, tLoopEnd, tWriteFileStart, tWriteFileEnd;
    timestamp_t tNeighbour, tCompute, tUpdate, tRead, tWriteFile, tInit, tLoop;
    tNeighbour = tCompute = tUpdate = tRead = tWriteFile = tInit = tLoop = 0;

    cl_event writeDone, neighbourDone, computeDone, readDone;
    cl_int status;

    t0 = get_timestamp();
    if(EXTRA_TIMING){
        tInitStart = get_timestamp();    
    }

    simTime = SIMTIME; // in miliseconds
    simSteps = ceil(simTime/DELTA);

    DEBUG_PRINT(("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2));

    //Open output file
    pOutFile = fopen(outFileName,"w");
    if(pOutFile==NULL){
        printf("Error: Couldn't create %s\n", outFileName);
        exit(EXIT_FAILURE);
    }
    writeOutput(temp, ("#simSteps Time(ms) Input(Iapp) Output(V_axon)\n"), pOutFile);



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

    FILE *computeFile, *neighbourFile;
    neighbourFile = fopen(neighbourFileName, "r");
    if(neighbourFile == NULL){
        printf("cannot open neighbour file\n");
        printf("current path: %s\n", neighbourFileName);
        exit(EXIT_FAILURE);
    }

    computeFile = fopen(computeFileName, "r");
    if(computeFile == NULL){
        printf("cannot open compute file\n");
        printf("current path: %s\n", computeFileName);
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
        sprintf(temp, "%d %.2f %.1f ", i+1, i*0.05, iApp); // start @ 1 because skipping initial values
        fputs(temp, pOutFile);

        if(EXTRA_TIMING){
            tNeighbourStart = get_timestamp();
        }

        //-----------------------------------------------------
        // STEP 11.1: Run neighbour kernel
        //-----------------------------------------------------


        if(EXTRA_TIMING){
            status = clWaitForEvents(1, &neighbourDone);
            tNeighbourEnd = get_timestamp();
            tNeighbour += (tNeighbourEnd - tNeighbourStart);
            tComputeStart = get_timestamp(); 
        }

        //-----------------------------------------------------
        // STEP 11.2: Run compute kernel
        //-----------------------------------------------------

        if(EXTRA_TIMING){
            status = clWaitForEvents(1, &computeDone);
            tComputeEnd = get_timestamp();
            tCompute += (tComputeEnd - tComputeStart);
            tReadStart = get_timestamp();
        }

        if(status != CL_SUCCESS){
            printf("error in loop, compute\n");
            exit(EXIT_FAILURE);
        }

        
        // transfer data from device to CPU
        if(WRITE_OUTPUT){
            //-----------------------------------------------------
            // STEP 11.3: Read output data from device
            //-----------------------------------------------------
        }

        if(EXTRA_TIMING){
            tReadEnd = get_timestamp();
            tRead += (tReadEnd - tReadStart);
        }

        // write output to file
        if(WRITE_OUTPUT){
            if(EXTRA_TIMING){
                tWriteFileStart = get_timestamp();
            }
            for(j = 0; j < IO_NETWORK_DIM1; j++){
                for(k = 0; k < IO_NETWORK_DIM2; k++){
                    writeOutputDouble(temp, cellStatePtr[((((i%2)^1)*IO_NETWORK_SIZE + (j+ k*IO_NETWORK_DIM1))* STATE_SIZE)+ AXON_V], pOutFile);
                }                
            }

            writeOutput(temp, ("\n"), pOutFile);
            if(EXTRA_TIMING){
                tWriteFileEnd = get_timestamp();
                tWriteFile += (tWriteFileEnd - tWriteFileStart);
            }
        }
    }
    if(EXTRA_TIMING){
        tLoopEnd = get_timestamp();
    }
    
    t1 = get_timestamp();
    usecs = (t1 - t0);// / 1000000;
    DEBUG_PRINT(("%d ms of brain time in %d simulation steps\n", simTime, simSteps));
    DEBUG_PRINT((" %lld usecs real time \n", usecs));

    if(EXTRA_TIMING){
        tInit = (tInitEnd - tInitStart);
        tLoop = (tLoopEnd - tLoopStart);
        
        DEBUG_PRINT(("\n"));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tInit: \t\t %lld \n", tInit));
        DEBUG_PRINT(("tLoop: \t\t %lld \n", tLoop));
        DEBUG_PRINT(("\ttNeighbour: \t %lld \n", tNeighbour));
        DEBUG_PRINT(("\ttCompute: \t %lld \n", tCompute));
        DEBUG_PRINT(("\ttRead: \t\t %lld \n", tRead));
        DEBUG_PRINT(("\ttWriteFile: \t %lld \n", tWriteFile));
        DEBUG_PRINT(("\t----------- + \n"));
        DEBUG_PRINT(("\ttSumLoop: \t %lld \n", (tWriteFile + tCompute + tNeighbour + tRead)));
        DEBUG_PRINT(("----------------------------------\n"));
        DEBUG_PRINT(("tSum: \t %lld \n", (tInit + tLoop)));
    }
    
    //-----------------------------------------------------
    // STEP 12: Release OpenCL resources
    //----------------------------------------------------- 


    //Free up memory and close files
    free(cellStatePtr);
    free(cellCompParamsPtr);
    fclose (pOutFile);

    return EXIT_SUCCESS;
}

