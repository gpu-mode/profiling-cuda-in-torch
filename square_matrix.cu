#include <stdio.h>

__global__ void square_matrix(float *matrix, float *result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

int main() {
    const int width = 10;  // Smaller width for demonstration
    const int height = 10; // Smaller height for demonstration
    const int size = width * height * sizeof(float);

    float *h_matrix = (float *)malloc(size);
    float *h_result = (float *)malloc(size);

    // Initialize and print the original matrix
    printf("Original Matrix:\n");
    for (int i = 0; i < width * height; ++i) {
        h_matrix[i] = 1.0f + i;  // Incremental values for demonstration
        printf("%f ", h_matrix[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }

    float *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, size);
    cudaMalloc((void **)&d_result, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    square_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_result, width, height);
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    // Print the squared matrix
    printf("\nSquared Matrix:\n");
    for (int i = 0; i < width * height; ++i) {
        printf("%f ", h_result[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }

    cudaFree(d_matrix);
    cudaFree(d_result);
    free(h_matrix);
    free(h_result);

    return 0;
}
