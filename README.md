## Misc notes and scripts
Used these for the first lecture of the CUDA mode series

Most of them are about profiling or authoring kernels in various GPU programming languages

# CUDA context init
Why does it take so long
1. Load CUDA drivers
2. Initialize memory management
3. Kernel compilation 
4. Device query selection: find available devices and check their capabilities
