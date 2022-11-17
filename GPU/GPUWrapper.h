#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define PROGRAM_FILE "matmult.cl"
#define KERNEL_FUNC "matmult"

#include <string>
#include <CL/cl.h>
#include <sstream>


namespace GPU {


	class DeviceWrapper {
		public:
			DeviceWrapper(cl_platform_id &platform);
			~DeviceWrapper();
            void AddParametrInProgram(cl_uint arg_index, const std::size_t size, void* data);
            cl_kernel& GetKernel() { return _kernel; }
            cl_command_queue& GetQueue() { return _queue; }
            cl_context& GetContext() { return _context; }
		private:
			void LoadProgramInDevice(const std::string& program_file, cl_int& err);

			cl_context _context;
			cl_program _program;
			cl_kernel _kernel;
			cl_device_id _device;
            cl_command_queue _queue;	
	};







}


