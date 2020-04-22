<h1 align='center'>Nova</h1>
<h3 align='center'>High performance GPU accelerated ray tracer using OpenCL/CUDA</h3>

## Examples
<p align="center">
  <img src="examples/fireplace.jpg" alt="fireplace" />
</p>
<p align="center">
  <img src="examples/dragon.jpg" alt="Dragon" />
</p>
<p align="center">
  <img src="examples/aircraft.jpg" alt="aircraft" />
</p>


## Dependencies For Building
* C++17
* CMake 3.12+
* OpenCL C++ or CUDA
* OpenMP
* OpenGL 3.3+
* GLFW dependencies (https://www.glfw.org/docs/3.3/compile.html#compile_deps)

## Build and Run

```bash
$ git clone https://github.com/wchang22/Nova.git
$ cd Nova && mkdir build && cd build
$ cmake .. -DBACKEND=OpenCL # or cmake .. -DBACKEND=CUDA
$ cmake --build .
$ ./nova
```

## Docker (For OpenCL)

```bash
$ cd Nova
$ docker build -t nova .
$ xhost local:root
$ docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/root/Nova --device /dev/dri:/dev/dri nova
$ cd Nova && mkdir build && cd build
$ cmake .. -DBACKEND=OpenCL
$ cmake --build .
$ ./nova
```

