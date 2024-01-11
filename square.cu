cuda_kernel = """
extern "C" __global__
void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = input[index] * input[index];
    }
}
"""