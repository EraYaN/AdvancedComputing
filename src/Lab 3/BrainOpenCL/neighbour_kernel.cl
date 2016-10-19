#include "kernel.h"

/**
Input: cellCompParamsPtr, cellStatePtr, i
cellCompParamsPtr: Array of struct which stores values of neighbours for each cell.
cellStatePtr: Array with values for each cell.
i: current simulation step


Retreive the voltage of the dendrite (V_dend) from each neighbour
**/
//int dev_fetch(int j, int k) {
//	return (j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE);
//}
//
//static double fetch_double(texture<int2, 2, cudaReadModeElementType> t, int x, int y) {
//	int2 v = tex2D(t, x, y);
//	return __hiloint2double(v.y, v.x);
//}

__kernel void neighbor_kernel(global user_float_t *cellStatePtr,global user_float_t *cellCompParamsPtr,  uint i){
	int j, k, n, p, q;

	k = get_global_id(0);
	j = get_global_id(1);

	//Get neighbor V_dend
	n = 0;
	for (p = j - 1; p <= j + 1; p++) {
		for (q = k - 1; q <= k + 1; q++) {
			//if (((p != j) || (q != k)) && ((p >= 0) && (q >= 0)) && ((p < IO_NETWORK_DIM1) && (q < IO_NETWORK_DIM2))) {
			//	cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i % 2][p][q].dend.V_dend;
			//} else if (p == j && q == k) {
			//	;   // do nothing, this is the cell itself
			//} else {
			//	//store same V_dend so that Ic becomes zero by the subtraction
			//	cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i % 2][j][k].dend.V_dend;
			//}
			//cellStatePtr[dev_fetch(j, k) + (n++)] = fetch_double(cellCompParamsPtr, p, q);
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

