// specify thread and block dimensions
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

// allocate GPU memory
unsigned char *dev_a, *dev_b;

//checkCudaCall(cudaHostGetDevicePointer(&dev_a, inputImage, 0));
//checkCudaCall(cudaHostGetDevicePointer(&dev_b, grayImage, 0));

checkCudaCall(cudaMalloc(&dev_a, 3*width*height * sizeof(unsigned char)));
checkCudaCall(cudaMalloc(&dev_b, width*height * sizeof(unsigned char)));

checkCudaCall(cudaMemcpy(dev_a, inputImage, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

auto t_kernel = now();
// execute actual function
rgb2grayCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, width, height);
//checkCudaCall(cudaThreadSynchronize());
auto t_cleanup = now();

checkCudaCall(cudaMemcpy(grayImage, dev_b, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

checkCudaCall(cudaFree(dev_a));
checkCudaCall(cudaFree(dev_b));