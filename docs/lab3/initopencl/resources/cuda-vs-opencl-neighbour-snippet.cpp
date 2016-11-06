// CUDA
k = blockIdx.x*blockDim.x + threadIdx.x;
j = blockIdx.y*blockDim.y + threadIdx.y;

...

__syncthreads();


// OpenCL equivalent
k = get_global_id(0);
j = get_global_id(1);

...

barrier(CLK_GLOBAL_MEM_FENCE);