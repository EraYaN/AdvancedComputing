// specify thread and block dimensions
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);