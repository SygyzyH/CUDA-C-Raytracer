# CUDA-C-Raytracer
A GPU accelerated raytracer written in CUDA C.

## Requirements
- Any NVIDIA GPU (If CUDA Toolkit is above version 6.0)
- CUDA Toolkit

CPU emulation support was removed since CUDA Toolkit 6.0. The program should support devices with compute capability 1.3 and above. The compute capability and the first device found are printed at the initialization of the program. 
