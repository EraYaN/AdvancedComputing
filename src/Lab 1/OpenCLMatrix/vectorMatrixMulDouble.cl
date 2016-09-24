__kernel void mul_kernel(global double *matrix1_in, global double *matrix2_in, global double *matrix_out, int size) {
	//Get the global id's
	int id_x = get_global_id(0);
	int id_y = get_global_id(1);
	double value = 0;

	//Do the actual math
	for (unsigned k = 0; k < size; k++) {
		value += matrix1_in[id_y * size + k] * matrix2_in[k * size + id_x];
	}

	//write back result
	matrix_out[id_y * size + id_x] = value;
}
