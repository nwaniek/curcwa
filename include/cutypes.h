#ifndef __CUTYPES_H__F58532B1_54F4_4310_AD8F_7746B729350E
#define __CUTYPES_H__F58532B1_54F4_4310_AD8F_7746B729350E

#include <cuda.h>
#include <cublas.h>
#define CULA_USE_CUDA_COMPLEX
#include <cula.h>

// types to be used by cuda and cpp files

typedef struct slice_entry
{
	unsigned nsteps;
	unsigned offset;
	float d;
} slice_entry_t;


typedef struct step_entry
{
	float x;	// x position
	unsigned m;	// material index
} step_entry_t;


__host__ __device__ inline void
swap(step_entry_t &a, step_entry_t &b)
{
	float x = a.x;
	unsigned m = a.m;

	a.x = b.x;
	b.x = x;

	a.m = b.m;
	b.m = m;
}


// structure to store device pointers
typedef struct dev_mem_ptr {
	// temporary memory required by cuda_full
	cuFloatComplex *full_A;
	cuFloatComplex *full_B;
	cuFloatComplex *full_lambda;
	cuFloatComplex *full_v;
	cuFloatComplex *full_vl;
	cuFloatComplex *full_vx;
	cuFloatComplex *full_vlx;

	// temporary memory required by secular
	cuFloatComplex *secular_A;
	cuFloatComplex *secular_C;
	cuFloatComplex *secular_epsmat;
	cuFloatComplex *secular_epsinv;

	// temporary memory used by inveps
	cuFloatComplex *inveps_A;
	culaDeviceInt *inveps_ipiv;
} dev_mem_ptr_t;




// get the device pointers struct
void release_dev_mem_ptr(dev_mem_ptr_t *dev_mem);


#endif /* __CUTYPES_H__F58532B1_54F4_4310_AD8F_7746B729350E */

