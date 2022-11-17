project(gpu-matcher)
cmake_minimum_required(VERSION 3.10)
add_compile_definitions(_DEVICE_TYPE_=CL_DEVICE_TYPE_CPU)

set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(matrixmult 
	"main.cpp"
    "GPU/GPUWrapper.cpp"
)

target_link_libraries(matrixmult OpenCL)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/GPU/matmult.cl
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
