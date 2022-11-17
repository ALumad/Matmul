#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdlib.h>
#include <sys/types.h>
#include <CL/cl.h>
#include <sstream>
#include <iostream>
#include <ctime>
#include "define.h"
#include "GPU/GPUWrapper.h"
#include <time.h>

int main() {
    std::cout << "Matrix dimension: " << core::MATRIX_SIZE << std::endl;
    
    cl_platform_id platform;
    cl_int err;
    
    float *A = new float[core::MATRIX_SIZE*core::MATRIX_SIZE];
    float *B = new float[core::MATRIX_SIZE*core::MATRIX_SIZE];
    float *res = new float[core::MATRIX_SIZE*core::MATRIX_SIZE];

    std::cout << "Matrix initialization" << std::endl;
    srand((unsigned int)time(0));
    for(int i = 0; i < core::MATRIX_SIZE*core::MATRIX_SIZE; i++) {
        A[i] = (float)rand()/RAND_MAX;
        B[i] = (float)rand()/RAND_MAX;

    }  
    
    clGetPlatformIDs(1, &platform, NULL);

    GPU::DeviceWrapper dwrapper(platform);
    cl_kernel& kernel = dwrapper.GetKernel();
    cl_command_queue& queue = dwrapper.GetQueue(); 
    cl_context context = dwrapper.GetContext();
    std::cout << "Device arguments initialization" << std::endl;
    cl_mem a_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, core::FULL_SIZE, A, &err);
    cl_mem b_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  core::FULL_SIZE, B, &err);    
    cl_mem res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, core::FULL_SIZE, NULL, &err);
    std::cout << "Add arguments" << std::endl;
    dwrapper.AddParametrInProgram(0, sizeof(cl_mem), &a_buff);
    dwrapper.AddParametrInProgram(1, sizeof(cl_mem), &b_buff);    
    dwrapper.AddParametrInProgram(2, sizeof(cl_mem), &res_buff); 

    std::cout << "Run algorithm" << std::endl;
    const clock_t begin_time = clock();
    size_t global_size = core::MATRIX_SIZE;
    OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL));
    OCL_SAFE_CALL(clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, core::FULL_SIZE, res, 0, NULL, NULL));

    std::cout << std::endl;
    std::cout << "time execute algorithm and read buffer: " <<  float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    clReleaseMemObject(a_buff);
    clReleaseMemObject(b_buff);
    clReleaseMemObject(res_buff);
    delete [] res;
    delete [] A;
    delete [] B;

    return 0;
}

