#ifndef __UTIL_HPP__6BB3D15B_098B_4767_9448_726A78B2C2C9
#define __UTIL_HPP__6BB3D15B_098B_4767_9448_726A78B2C2C9

#include <iostream>
#include <cublas.h>
#define CULA_USE_CUDA_COMPLEX
#include <cula.h>
#include "types.hpp"


/*
 * call guard for cula functions. will exit on failure
 */
#define CULA_GUARD(f, ...) { \
	culaStatus status = f(__VA_ARGS__); \
	if (status != culaNoError) { \
		std::cout << "CULA ERROR while calling " # f <<  ", line: " << \
		    __LINE__ << ", file: " << __FILE__ << ", code: " << status \
		    << " (" << culaGetStatusString(status);  \
		if (status == culaDataError) \
			std::cout << ": code = " << culaGetErrorInfo(); \
		std::cout << ")" << std::endl; \
		exit(EXIT_FAILURE); \
	}}


/*
 * call guard for cublas functions. will exit on failure
 */
#define CUBLAS_GUARD(f, ...) { \
	cublasStatus status = f(__VA_ARGS__); \
	if (status != CUBLAS_STATUS_SUCCESS) { \
		std::cout << "CUDA ERROR while calling " # f <<  ", line: " << \
		    __LINE__ << ", file: " << __FILE__ << ", code: " << status \
		    << std::endl; \
		exit(EXIT_FAILURE); \
	}}


/*
 * Guard for cublas using the cublasGetError function
 */
#define CUBLAS_GUARD_E(f, ...) { \
	f(__VA_ARGS__); \
	cublasStatus status = cublasGetError(); \
	if (status != CUBLAS_STATUS_SUCCESS) { \
		std::cout << "CUDA ERROR while calling " # f <<  ", line: " << \
		    __LINE__ << ", file: " << __FILE__ << ", code: " << status \
		    << std::endl; \
		exit(EXIT_FAILURE); \
	}}


/*
 * call guard for cuda functions. will exit on failure
 */
#define CUDA_GUARD(f, ...) { \
	cudaError_t err = f(__VA_ARGS__); \
	if (err != cudaSuccess) { \
		std::cout << "CUDA ERROR while calling " # f <<  ", line: " << \
		    __LINE__ << ", file: " << __FILE__ << ", code: " << err \
		    << " (" << cudaGetErrorString(err) << ")" << std::endl; \
		exit(EXIT_FAILURE); \
	}}

#define CUDA_CHECK(fname, r) { \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) { \
		std::cout << "CUDA ERROR while calling " # fname << ", line: " << \
		    (__LINE__ - r) << ", file: " << __FILE__ << ", code: " << err \
		    << " (" << cudaGetErrorString(err) << ")" << std::endl; \
		exit(EXIT_FAILURE); \
	}}


/*
 * macro for being able to index like you would in fortran, index starting at 1
 * and not at 0
 */
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))


/*
 * regular indexing macro to index the (i,j)-th element in an MxN matrix that is
 * stored in col-major order. ld is the number of rows
 */
#define IDX(i,j,ld) ((j) * (ld) + (i))

/*
 * indexing macro to index stuff that is stored row-major. ld is the number of
 * elements per row
 */
#define IDXR(i, j, ld) ((i) * (ld) + (j))


/**
 * linterpol - linearly interpolate refractive indices for a wavelength.
 *
 * linearly interpolate the refractive index of a material for a wavelength wl.
 * if the material has only one specified wavelength, the corresponding
 * refractive index will be returned.
 *
 * m:	the material that holds the refractive indices
 * wl:	the wavelength the refractive index shall be interpolated for
 */
cuFloatComplex linterpol(const material *m, const float wl);



#endif /* __UTIL_HPP__6BB3D15B_098B_4767_9448_726A78B2C2C9 */

