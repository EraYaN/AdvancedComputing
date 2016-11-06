#include "kernel.h"

/**
Input: cellVDendPtr, cellStatePtr, i
cellVDendPtr: Array of struct which stores values of neighbours for each cell.
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

inline int dev_fetch(int j, int k) {
	return (j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE);
}

inline int dev_fetch_vdend(int p, int q) {
	return (p*IO_NETWORK_DIM1 + q);
}

inline double fetch_double(image2d_t t, sampler_t sampler, int x, int y) {
	return as_double2(read_imageui(t, sampler, (int2)(x, y))).x;
	//return d2.x;
	//return 0;
}

__kernel void neighbour_kernel(global user_float_t *cellStatePtr, __read_only image2d_t cellVDendPtr) {
	int j, k, n, p, q;

	k = get_global_id(0);
	j = get_global_id(1);

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

	//get neighbor v_dend
	n = 0;
	for (p = j - 1; p <= j + 1; p++) {
		for (q = k - 1; q <= k + 1; q++) {
			//if (((p != j) || (q != k)) && ((p >= 0) && (q >= 0)) && ((p < io_network_dim1) && (q < io_network_dim2))) {
			//	cellvdendptr[j][k].neighvdend[n++] = cellstateptr[i % 2][p][q].dend.v_dend;
			//} else if (p == j && q == k) {
			//	;   // do nothing, this is the cell itself
			//} else {
			//	//store same v_dend so that ic becomes zero by the subtraction
			//	cellvdendptr[j][k].neighvdend[n++] = cellstateptr[i % 2][j][k].dend.v_dend;
			//}

			/*if (((p != j) || (q != k)) && ((p >= 0) && (q >= 0)) && ((p < io_network_dim1) && (q < io_network_dim2))) {*/
				//printf("k,j : %d, %d\ndev_fetch(j, k): %d\ndev_fetch_vdend(p, q): %d\nn: %d\nvdend: %lf\n", k, j, dev_fetch(j, k), dev_fetch_vdend(p, q), n, cellvdendptr[dev_fetch_vdend(p, q)]);
			cellStatePtr[dev_fetch(j, k) + (n++)] = fetch_double(cellVDendPtr, sampler, p, q);
			/*} else if (p == j && q == k) {
				//	;   // do nothing, this is the cell itself
			} else {
				//printf("k,j : %d, %d\ndev_fetch(j, k): %d\ndev_fetch_vdend(j, k): %d\nn: %d\nvdend: %lf\n", k, j, dev_fetch(j, k), dev_fetch_vdend(j, k), n, cellvdendptr[dev_fetch_vdend(j, k)]);
				cellstateptr[dev_fetch(j, k) + (n++)] = cellvdendptr[fetch_double(j, k)];
			}*/

			if (p == j && q == k) n = n - 1;
		}
	}

	/*for (int ind = 0; ind < io_network_size; ind++) {
		for (int b = 0; b < param_size; b++) {
			printf("(state_n) %d,%d: %lf\n", ind, b, cellstateptr[ind*param_size + b]);
		}
	}*/

	barrier(CLK_GLOBAL_MEM_FENCE);
}

