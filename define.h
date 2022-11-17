#pragma once
#include <CL/cl.h>
#include <string>
#include <exception>
#include <sstream>


void inline reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;



    std::ostringstream errss, liness;
    errss << err;
    liness << line;

    std::string message = "OpenCL error code " + errss.str() + " encountered at " + filename + ":" + liness.str();
    throw std::runtime_error(message);
}


#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)



namespace core {
    const int MATRIX_SIZE = 10000;
    const size_t FULL_SIZE = sizeof(float)*MATRIX_SIZE*MATRIX_SIZE;
};

