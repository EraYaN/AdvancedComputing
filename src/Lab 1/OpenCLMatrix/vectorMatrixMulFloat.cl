__kernel void mul_kernel(global float *vector_in, global float *matrix_in, global float *vector_out, int size){
  int id = get_global_id(0);
  float value = 0;
  int k;

  for(k = 0; k < size; k++){
    value += matrix_in[(k* size) + id] * vector_in[k];
  }

  vector_out[id] = value;

}
