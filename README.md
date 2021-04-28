# CUDA-C-Raytracer
A GPU accelerated raytracer written in CUDA C.
![alt text](https://github.com/SygyzyH/CUDA-C-Raytracer/blob/main/res/Thumbnail.PNG?raw=true)

## Requirements
- CUDA Toolkit
- Any NVIDIA GPU (If CUDA Toolkit is above version 6.0)

CPU emulation support was removed since CUDA Toolkit 6.0, therefore if the toolkit version is above 6.0 a physical NVIDIA GPU is required. The program should support devices with compute capability 1.3 and above. The compute capability and the first compatible device found are printed at the initialization of the program. 

## Compilation and execution
With the CUDA Toolkit installed, run the instructions in the terminal, in the same directory the project is installed:
```
nvcc rtapi.cu view.cpp manager.cpp -o view -lgdi32 -luser32 && view.exe
```
This will create the file:
```
view.exe
```
And run it in the terminal.
```view.exe``` can be deleted later.

## Using the program
If all the requirements are met, a new window will open, displaying the current output of the raytracer.
To move the camera around, use the arrow keys. To move up and down, use space and control respectively.
To rotate the camera, use the mouse.

To quit, either close the window or press ESC and accept the prompt.

## TODO
This project is far from being complete. Here is a list of the features i think would be nice to have:
- [X] Diffused lighting
- [X] Specular lighting
- [X] Triangles
- [X] Polygon mesh
- [X] .obj file support
- [ ] Texturing
- [ ] Proper loading of CUDA Runtime API DLL and error detection
- [X] API for C++ interface
- [X] C++ Windows API level display
