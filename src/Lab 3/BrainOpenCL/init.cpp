#include "init.h" 

void mallocCells(user_float_t **cellVDendPtr, user_float_t **cellStatePtr){
    int k;
    DEBUG_PRINT(("cellStatePtr: %lu B\n", IO_NETWORK_SIZE*PARAM_SIZE *sizeof(user_float_t)));
    //Two cell state structs are needed so as to avoid having to synchronize all consumers before they start rewriting the cell state.
    (*cellStatePtr) = (user_float_t*)malloc(IO_NETWORK_SIZE*PARAM_SIZE *sizeof(user_float_t));//current and next state
    if((*cellStatePtr)==NULL){
        printf("Error: Couldn't malloc for cellStatePtr\n");
        exit(EXIT_FAILURE);
    }	

    DEBUG_PRINT(("cellVDendPtr: %lu B\n", IO_NETWORK_SIZE*sizeof(user_float_t)));
    (*cellVDendPtr) = (user_float_t*)malloc(IO_NETWORK_SIZE*sizeof(user_float_t));
    if((*cellVDendPtr) ==NULL){
        printf("Error: Couldn't malloc for cellVDendPtr\n");
        exit(EXIT_FAILURE);
    }
}

void InitState(user_float_t *cellStatePtr, user_float_t *cellVDendPtr){
    int i, b;
    cl_float_t  cellStateInit[STATE_SIZE];
    //Initial dendritic parameters
    cellStateInit[DEND_V]   = -60;  
    cellStateInit[DEND_H]   = 0.0337836;
    cellStateInit[DEND_CAL] = 0.0112788;
    cellStateInit[DEND_P]   = 0.0049291;
    cellStateInit[DEND_I]   = 0.5;
    cellStateInit[DEND_CA2] = 3.7152;

    cellStateInit[SOMA_G]   = 0.68;
    cellStateInit[SOMA_V]   = -60;
    cellStateInit[SOMA_SM]  = 1.0127807;
    cellStateInit[SOMA_SH]  = 0.3596066;
    cellStateInit[SOMA_CK]  = 0.7423159;
    cellStateInit[SOMA_CL]  = 0.0321349;
    cellStateInit[SOMA_PN]  = 0.2369847;
    cellStateInit[SOMA_PP]  = 0.2369847;
    cellStateInit[SOMA_PXS] = 0.1;

    cellStateInit[AXON_V]   = -60;
    cellStateInit[AXON_SM]  = 0.003596066;
    cellStateInit[AXON_SH]  = 0.9;
    cellStateInit[AXON_P]   = 0.2369847;

    //Copy init sate to all cell states
    for(i=0;i<IO_NETWORK_SIZE;i++){
        for(b=0;b<STATE_SIZE;b++){
            cellStatePtr[i*PARAM_SIZE + b + STATEADD] = cellStateInit[b];
        }
		cellVDendPtr[i] = cellStateInit[DEND_V];
    }

    return;
}

void init_g_CaL(user_float_t *cellStatePtr){
    int seedvar, i;
    seedvar = 1;
    for(i=0;i<IO_NETWORK_SIZE;i++){
            srand(seedvar++);   // use this for debugging, now there is difference
            //cellStatePtr[(IO_NETWORK_SIZE + i)*STATE_SIZE + SOMA_G] = cellStatePtr[i*STATE_SIZE + SOMA_G] = 0.68;
			cellStatePtr[i*PARAM_SIZE + SOMA_G + STATEADD] = 0.68;
    }
}
