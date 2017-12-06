#define BLOCK_SIZE 16

__global__ void MatMulKernel(float* A, float* B, float* C, int width)
{
    __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;
	
	float PValue = 0;
	
	for(int m = 0; m < width/BLOCK_SIZE; m++)
	{
	   Ads[ty][tx] = A[row * width + m*BLOCK_SIZE + tx];
	   Bds[ty][tx] = B[(m*BLOCK_SIZE+ty) * width + col];
	   __syncthreads();
	   
	   for(int k = 0; k < BLOCK_SIZE; k++)
	   {
	     PValue += Ads[ty][k]*Bds[k][tx];
	   }
	   __syncthreads();
	}
	C[row*width+col] = PValue;
}


void MatMul(const float* A, const float* B, float* C, int width_A, int height_A, int width_B, int width_B)
{

    float* d_A;
	size_t size = sizeof(A);
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	
	size = sizeof(B);
	float* d_B;
	cudaMalloc(&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	
	float* d_C;
	size_t size_C = height_A*width_B*sizeof(float);
	cudaMalloc(&d_C, size_C);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(width_B/BLOCK_SIZE, height_A/BLOCK_SIZE);
	
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width_A);
	
	cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
}