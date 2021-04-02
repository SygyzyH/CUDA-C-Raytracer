# CUDA-C-Raytracer
A GPU accelerated raytracer written in CUDA C.

## Requirements
- CUDA Toolkit
- Any NVIDIA GPU (If CUDA Toolkit is above version 6.0)

CPU emulation support was removed since CUDA Toolkit 6.0, therefore if the toolkit version is above 6.0 a physical NVIDIA GPU is required. The program should support devices with compute capability 1.3 and above. The compute capability and the first compatible device found are printed at the initialization of the program. 

## Compilation and execution
With the CUDA Toolkit installed, run the instructions in the terminal, in the same directory the project is installed:
```
nvcc main.cu
```
This will create three files:
```
a.exe
a.exp
a.lib
```
And to run the program, run the executable ```a.exe```, preferably in the terminal.
Since the project is yet to be properly displayed, the resulting pixel colors will be printed in the terminal. Therefore, it is reccomended to run the program in some IDE or text editing envirment, to make sense of the output.
The three files created can later be deleted.

## TODO
This project is far from being complete. Here is a list of the features i think would be nice to have:
- [ ] Diffused lighting
- [ ] Specular lighting
- [ ] Triangles
- [ ] Polygon mesh
- [ ] .obj file support
- [ ] Texturing
- [ ] Proper loading of CUDA Runtime API DLL and error detection
- [ ] API for C++ interface
- [ ] C++ Windows API level display
