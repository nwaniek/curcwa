#include "cu/wavevec.h"
#include "util.hpp"
#include <iostream>
#include "cmplxutil.hpp"
#include "global.h"

/**
 * void d_wavevec - calculate the wave vectors for reflectance and transmission.
 *
 * Assumes that the first and the last slice are made up of just one material.
 *
 */
__global__ void
d_wavevec(unsigned nslices, slice_entry_t *slices, cuFloatComplex *refrac,
	order_t order, const float T, float *wavelengths, step_entry_t *steps,
	float theta, float *q, cuFloatComplex *kt, cuFloatComplex *kr,
	const unsigned nloops, const unsigned norders, unsigned nmats)
{
	// diffraction order index to calculate result for
	unsigned i = threadIdx.x;

	// wavelength index to get the proper wavelength
	const unsigned wi = blockIdx.x;

	// number of wavelengths stored in matrices. required for index calc.
	const unsigned nwavelengths = gridDim.x;

	// actual wavelength
	const float wl = wavelengths[wi];
	const float Q = wl / T;

	const slice_entry_t slice0 = slices[0];
	const slice_entry_t sliceN = slices[nslices - 1];

	// material indices to get the refractive index for specific wavelength
	// wl
	const unsigned mat0 = steps[slice0.offset].m;
	const unsigned matN = steps[sliceN.offset].m;

	// TODO: check if the following is proper math
	// in the fortran code, eps_r as well as r are handled real-only.
	// therefore, just take the real part

	for (unsigned j = 0; j < nloops; j++, i += blockDim.x) {
		if (i >= norders)
			return;

		const float eps_r = refrac[IDX(mat0, i, nmats)].x;
		const cuFloatComplex eps_t = refrac[IDX(matN, i, nmats)];
		const float r = sqrt(eps_r) * sin(theta);

		const float qi = r + i * Q;

		cuFloatComplex kri = {eps_r - qi*qi, 0.0f};
		csqrti(kri);
		if (kri.y < 0.0f)
			kri *= -1.0f;

		cuFloatComplex kti = {qi * qi, 0.0f};
		kti = eps_t - kti;
		csqrti(kti);
		if (kti.y < 0.0f)
			kti *= -1.0f;

		// write back
		__syncthreads();
#if ROW_MAJOR
		q[IDXR(wi, i, nwavelengths)] = qi;
		kt[IDXR(wi, i, nwavelengths)] = kti;
		kr[IDXR(wi, i, nwavelengths)] = kri;
#else
		q[IDX(wi, i, nwavelengths)] = qi;
		kt[IDX(wi, i, nwavelengths)] = kti;
		kr[IDX(wi, i, nwavelengths)] = kri;
#endif
	}
}


/*
 *
 *
 * dev_q:	output -> wave vectors
 * dev_kt:	output -> k_t vector matrix
 *
 */
extern "C"
void cuda_wavevec(unsigned nslices, slice_entry_t *dev_slices,
		cuFloatComplex *dev_refrac_idx, unsigned nwavelengths,
		unsigned nmats,
		float *dev_wavelength, order_t order, const float T,
		step_entry_t *dev_steps, float theta, float *dev_q,
		cuFloatComplex *dev_kt, cuFloatComplex *dev_kr)
{
	// required: order.lo, order.hi, incident angle, Q
	//   T: overall slice length
	//
	// slice length is one (the same) value for every slice, so pass it in
	// as parameter
	// storage space for q, kr, kt (per wavelength)
	dim3 blocks(nwavelengths);

	// problem when order's are too many:
	unsigned norders = order.hi - order.lo + 1;
	unsigned nloops = norders / 512 + 1;
	dim3 threads(min(norders, 512));

	d_wavevec<<<blocks, threads>>>(nslices, dev_slices, dev_refrac_idx,
			order, T, dev_wavelength, dev_steps, theta, dev_q,
			dev_kt, dev_kr, nloops, norders, nmats);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "ERROR: d_wavevec failed: " <<
			err << " (" << cudaGetErrorString(err) << ")" <<
			std::endl;
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
}
