# llama.cpp
branch dev-unify_profiler

## Fast Usage
```bash
chmod +x build-for-debug.sh && build-for-debug.sh
```

## Code Modify
```
├── common/
│   ├── arg.cpp
│   ├── common.cpp
│   └── common.h
├── docs/
│   └── cross-profiler.md
├── examples/
│   └── debug/
│       └── debug.cpp
├── ggml/
│   ├── include/
│   │   ├── ggml-cpu.h
│   │   └── ggml-profiler.h
│   └── src/
│       ├── CMakeLists.txt
│       ├── ggml-backend-impl.h
│       ├── ggml-backend.cpp
│       ├── ggml-blas/
│       │   └── ggml-blas.cpp
│       ├── ggml-cann/
│       │   └── ggml-cann.cpp
│       ├── ggml-cpu/
│       │   ├── ggml-cpu.c
│       │   └── ggml-cpu.cpp
│       ├── ggml-cuda/
│       │   ├── common.cuh
│       │   ├── ggml-cuda.cu
│       │   └── vendors/
│       │       ├── hip.h
│       │       └── musa.h
│       ├── ggml-hexagon/
│       │   └── ggml-hexagon.cpp
│       ├── ggml-metal/
│       │   ├── ggml-metal-context.h
│       │   ├── ggml-metal-context.m
│       │   ├── ggml-metal-device.h
│       │   ├── ggml-metal-device.m
│       │   ├── ggml-metal-ops.cpp
│       │   ├── ggml-metal-ops.h
│       │   └── ggml-metal.cpp
│       ├── ggml-opencl/
│       │   └── ggml-opencl.cpp
│       ├── ggml-openvino/
│       │   └── ggml-openvino.cpp
│       ├── ggml-profiler.cpp
│       ├── ggml-rpc/
│       │   └── ggml-rpc.cpp
│       ├── ggml-sycl/
│       │   └── ggml-sycl.cpp
│       ├── ggml-virtgpu/
│       │   └── ggml-backend.cpp
│       ├── ggml-vulkan/
│       │   ├── ggml-vulkan.cpp
│       │   └── vulkan-shaders/
│       │       └── types.glsl
│       ├── ggml-webgpu/
│       │   └── ggml-webgpu.cpp
│       ├── ggml-zdnn/
│       │   └── ggml-zdnn.cpp
│       └── ggml-zendnn/
│           └── ggml-zendnn.cpp
├── include/
│   └── llama.h
├── src/
│   └── llama-context.cpp
├── tests/
│   ├── export-graph-ops.cpp
│   └── test-backend-ops.cpp
└── tools/
    ├── cli/
    │   └── cli.cpp
    ├── completion/
    │   └── completion.cpp
    └── profiler/
        ├── __init__.py
        └── profiler.py
```