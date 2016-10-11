// initialize the thread numbers
int x = blockIdx.x * blockDim.x + threadIdx.x; // width
int y = blockIdx.y * blockDim.y + threadIdx.y; // height

if (x < width && y < height) {
	// CPU code in for loops
}