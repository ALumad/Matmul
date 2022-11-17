#include "GPUWrapper.h"
#include <cstring>
#include <string>
#include "define.h"
#include <CL/cl.h>
#include <iostream>
#ifndef _DEVICE_TYPE_
#define _DEVICE_TYPE_ CL_DEVICE_TYPE_GPU
#endif

namespace GPU {
	DeviceWrapper::DeviceWrapper(cl_platform_id &platform){
		OCL_SAFE_CALL(clGetDeviceIDs(platform, _DEVICE_TYPE_, 1, &_device, NULL));
		cl_int err;
		_context = clCreateContext(NULL, 1, &_device, NULL, NULL, &err);
		LoadProgramInDevice(PROGRAM_FILE, err);
        _kernel = clCreateKernel(_program, KERNEL_FUNC, &err);
        _queue = clCreateCommandQueue(_context, _device, 0, &err);
	}
    
    DeviceWrapper::~DeviceWrapper() {

        clReleaseCommandQueue(_queue);        
        clReleaseKernel(_kernel);
        clReleaseProgram(_program);
        clReleaseContext(_context);

    }


   void DeviceWrapper::AddParametrInProgram(cl_uint arg_index, const std::size_t size, void* data)  {
        OCL_SAFE_CALL(clSetKernelArg(_kernel, arg_index, size, data));
    }


    void DeviceWrapper::LoadProgramInDevice(const std::string& program_file, cl_int& err) {
        FILE *program_handle = fopen(program_file.c_str(), "r");
        fseek(program_handle, 0, SEEK_END);

        size_t program_size = ftell(program_handle);
        rewind(program_handle);

        char* program_buffer = (char*)malloc(program_size + 1);
        program_buffer[program_size] = '\0';

        fread(program_buffer, sizeof(char), program_size,  program_handle);
        fclose(program_handle);


        _program = clCreateProgramWithSource(_context, 1, (const char**)&program_buffer, &program_size, &err);

        free(program_buffer);

        clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
    }	

}




