# CUDA-C-Raytracer
A GPU accelerated raytracer written in CUDA C.

## Requirements
- CUDA Toolkit
- Any NVIDIA GPU (If CUDA Toolkit is above version 6.0)

CPU emulation support was removed since CUDA Toolkit 6.0, therefore if the toolkit version is above 6.0 a physical NVIDIA GPU is required. The program should support devices with compute capability 1.3 and above. The compute capability and the first compatible device found are printed at the initialization of the program. 

## TODO
This project is far from being complete.
- [ ] Diffused lighting
- [ ] Specular lighting
- [ ] Triangles
- [ ] Polygon mesh
- [ ] .obj file support
- [ ] Texturing
- [ ] Proper loading of CUDA Runtime API DLL and error detection
- [ ] API for C++ interface
- [ ] C++ Windows API level display
