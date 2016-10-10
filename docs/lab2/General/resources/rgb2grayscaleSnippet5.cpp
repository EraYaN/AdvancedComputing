// copy GPU memory to CPU memory
checkCudaCall(cudaMemcpy(grayImage, dev_b, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));