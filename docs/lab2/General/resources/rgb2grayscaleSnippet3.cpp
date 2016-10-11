// specify thread and block dimensions
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(ceil((double)width / threadsPerBlock.x), ceil((double)height / threadsPerBlock.y));