#include "cu/fouriercoeff.h"
#include "util.hpp"
#include <iostream>
#include "cmplxutil.hpp"


__global__ void
d_fouriercoeff_zero(slice_entry_t *slices, step_entry_t *steps,
		cuFloatComplex* coeffs, cuFloatComplex *refrac,
		const int norders, const unsigned nloops,
		const unsigned nwavelengths, const unsigned nmats)
{
	// calculating only one value
	cuFloatComplex c0 = {0.0, 0.0};

	// slice index
	const unsigned si = blockIdx.x;
	// wave-length-index. required for refractive index
	unsigned wi = threadIdx.x;

	// number of slices in this stack
	const unsigned nslices = gridDim.x;

	// get the slice map entry
	slice_entry_t slice = slices[si];

	step_entry_t stepn = steps[slice.offset + slice.nsteps - 1];
	const float T = stepn.x;

	/*
	 * this will possibly be heavily divergent when nsteps changes a lot
	 */
	for (unsigned j = 0; j < nloops; j++, wi += blockDim.x) {
		if (wi >= nwavelengths)
			return;

		step_entry_t step0 = steps[slice.offset];
		float x0 = step0.x;

		c0 = refrac[IDX(step0.m, wi, nmats)];
		c0 *= x0;

		for (unsigned i = 1; i < slice.nsteps; i++) {
			stepn = steps[slice.offset + 1];
			cuFloatComplex f = refrac[IDX(stepn.m, wi, nmats)] * (stepn.x - step0.x);
			c0 += f;
			step0 = stepn;
		}
		c0 *= (1/T);
		__syncthreads();
		const unsigned i = si + wi * nslices;
		coeffs[IDX(i, norders, nslices * nwavelengths)] = c0;
	}
}


__global__ void
d_fouriercoeff_nonzero(slice_entry_t *slices, step_entry_t *steps,
		cuFloatComplex* coeffs, cuFloatComplex *refrac,
		const unsigned norders, const unsigned nloops,
		const unsigned nmats)
{
	const float two_pi = 2 * M_PI;

	// coefficient to calculate
	unsigned ci = threadIdx.x;
	// slice index
	const unsigned si = blockIdx.x;
	// wave-length-index. required for refractive index
	const unsigned wi = blockIdx.y;
	// number of slices in this stack
	const unsigned nslices = gridDim.x;
	// number of wavelengths stored
	const unsigned nwavelengths = gridDim.y;
	const unsigned nrows = nslices * nwavelengths;

	// get the slice map entry and last step for overall-width
	slice_entry_t slice = slices[si];
	step_entry_t stepn = steps[slice.offset + slice.nsteps - 1];
	const float T = stepn.x;

	for (unsigned c = 0; c < nloops; c++, ci += blockDim.x) {
		if (ci >= (norders << 1))
			return;

		// calculation of coefficients
		step_entry_t step0  = steps[slice.offset];
		cuFloatComplex f0   = refrac[IDX(step0.m, wi, nmats)];
		cuFloatComplex f1   = refrac[IDX(stepn.m, wi, nmats)];
		cuFloatComplex tmp  = f1 - f0;
		cuFloatComplex tmp2 = {0.0f, 0.0f};

		// calculate k<>0 indices
		int k = ci - norders;
		if (k >= 0) ++k;

		for (unsigned i = 1; i < slice.nsteps; i++) {
			stepn = steps[slice.offset + i];

			// f0 = refrac[IDX(step0.m, wi, nwavelengths)];
			f1 = refrac[IDX(stepn.m, wi, nmats)];
			f0 = f0 - f1;

			tmp2.x = 0.0f;
			tmp2.y = -1.0f;
			tmp2 *= (two_pi * (float)k/T * step0.x);
			cexpi(tmp2);

			f0 *= tmp2;
			tmp += f0;

			step0 = stepn;

			// prevent fetching f1 twice from global memory
			f0 = f1;
		}
		f0.x = 0.0f;
		f0.y = 1.0f;
		f0 /= (two_pi * k);
		tmp *= f0;

		// write back: beware of k>0 !
		const unsigned j = k > 0 ? ci + 1 : ci;
		const unsigned i = si + wi * nslices;
		__syncthreads();
		coeffs[IDX(i, j, nrows)] = tmp;
	}
}


extern "C"
void cuda_fouriercoeff(unsigned n_wavelengths, cuFloatComplex *dev_refrac_idx,
	unsigned nslices, slice_entry_t *slice, step_entry_t *steps,
	int norders, cuFloatComplex *fouriercoeffs, unsigned nmats)
{
	/*
	 * first kernel computes the values that are not at c(0)
	 */
	unsigned nthreads = 2 * norders;
	unsigned nloops = (nthreads >> 8) + 1;
	nthreads = min(nthreads, 512);

	dim3 blocks(nslices, n_wavelengths);
	dim3 threads(nthreads);

	d_fouriercoeff_nonzero<<<blocks, threads>>>(slice, steps,
			fouriercoeffs, dev_refrac_idx, norders, nloops, nmats);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "ERROR: d_fouriercoeff_nonzero failed: " <<
			err << " (" << cudaGetErrorString(err) << ")" <<
			std::endl;
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();

	/*
	 * second kernel computes remaining values at position c(0) for every
	 * slice and wavelength
	 */
	nloops = (n_wavelengths >> 8) + 1;
	blocks = dim3(nslices);
	threads = dim3(min(n_wavelengths, 512));
	d_fouriercoeff_zero<<<blocks, threads>>>(slice, steps, fouriercoeffs,
			dev_refrac_idx, norders, nloops, n_wavelengths, nmats);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "ERROR: d_fouriercoeff_zero failed: " <<
			err << " (" << cudaGetErrorString(err) << ")" <<
			std::endl;
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
}
