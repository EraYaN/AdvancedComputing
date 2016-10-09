// allocate GPU memory
unsigned char *dev_a, *dev_b;

checkCudaCall(cudaMalloc(&dev_a, 3*width*height * sizeof(unsigned char)));
checkCudaCall(cudaMalloc(&dev_b, width*height * sizeof(unsigned char)));