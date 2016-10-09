// execute kernel
rgb2grayCudaKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, width, height);