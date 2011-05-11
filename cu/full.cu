#include "cu/full.h"
#include "util.hpp"
#include "cu/secular.h"
#include <cuComplex.h>
#include "cmplxutil.hpp"
#include "global.h"


__global__ void
d_full()
{

}


__global__ void
prepare_for_TE(unsigned nz, unsigned l, cuFloatComplex *a, cuFloatComplex *b,
		cuFloatComplex *kt, cuFloatComplex *kr, const unsigned nloops,
		const unsigned wli, const unsigned n,
		const unsigned nwavelengths, cuFloatComplex *eps,
		const unsigned eps_nrows, const unsigned nslices)
{

}

__global__ void
prepare_for_TM(unsigned nz, unsigned l, cuFloatComplex *a, cuFloatComplex *b,
		cuFloatComplex *kt, cuFloatComplex *kr, const unsigned nloops,
		const unsigned wli, const unsigned n,
		const unsigned nwavelengths, cuFloatComplex *eps,
		const unsigned eps_nrows, const unsigned nslices)
{
	const cuFloatComplex j = {.0f, 1.f};
	const cuFloatComplex one = {1.f, .0f};
	const unsigned eps_offset = nslices * wli;
	cuFloatComplex tmp = {.0f, .0f};

	cuFloatComplex eps0 = eps[IDX(eps_offset + 0, nz, eps_nrows)];
	cuFloatComplex epsL = eps[IDX(eps_offset + nslices - 1, nz, eps_nrows)]; // is the index right?!

	// first thread does single value setup.
	if (threadIdx.x == 0) {
		b[nz].x = 1.f;

		tmp = cuCdivf(j, eps0);
#if ROW_MAJOR
		b[nz + n] = tmp * kr[IDXR(wli, nz, nwavelengths)];
#else
		b[nz + n] = tmp * kr[IDX(wli, nz, nwavelengths)];
#endif
	}
	__syncthreads();


	const unsigned nn = 2 * n * (l + 1);
	unsigned i = threadIdx.x;
	for (unsigned k = 0; k < nloops; k++, i += blockDim.x) {
		if (i >= n)
			return;

		a[IDX(i, i, nn)] = one * -1.f;

		tmp = cuCdivf(j, eps0);
#if ROW_MAJOR
		a[IDX(n + i, i, nn)] = tmp * kr[IDXR(wli, i, nwavelengths)];
#else
		a[IDX(n + i, i, nn)] = tmp * kr[IDX(wli, i, nwavelengths)];
#endif

		unsigned idx = 2 * n * l + i - 1;
		a[IDX(idx, idx + n, nn)] = one;

		tmp = cuCdivf(j, epsL);
#if ROW_MAJOR
		a[IDX(idx + n, idx + n, nn)] = tmp * kt[IDXR(wli, i, nwavelengths)];
#else
		a[IDX(idx + n, idx + n, nn)] = tmp * kt[IDX(wli, i, nwavelengths)];
#endif
	}
}


extern "C"
void cuda_full(pol_t pol,
		order_t order,
		unsigned nwavelengths,
		unsigned nslices,
		cuFloatComplex *dev_kt,
		cuFloatComplex *dev_kr,
		float *dev_q,
		slice_entry_t *slices,
		unsigned nmats,
		cuFloatComplex *coeffs,
		dev_mem_ptr_t *dev_mem
	      )
{
	typedef void (*prep_f)(unsigned nz, unsigned l, cuFloatComplex *a,
			cuFloatComplex *b, cuFloatComplex *kt,
			cuFloatComplex *kr, const unsigned nloops,
			const unsigned int wli, const unsigned n,
			const unsigned nwavelengths, cuFloatComplex *eps,
			const unsigned eps_nrows, const unsigned nslices);
	static const prep_f prep[] = {prepare_for_TE, prepare_for_TM};

	const unsigned norders = order.hi - order.lo + 1;
	const unsigned l = nslices - 2;
	const unsigned nn = 2 * norders * (l + 1);
	// static const unsigned nz = -order.lo + 1;

	CUDA_GUARD(cudaMemset, dev_mem->full_A, 0, sizeof(cuFloatComplex) * nn * nn);
	CUDA_GUARD(cudaMemset, dev_mem->full_B, 0, sizeof(cuFloatComplex) * nn);

	// assume that there are usually less than 512 orders to be calculated,
	// therefore just run one block!
	unsigned nthreads = order.hi - order.lo + 1;
	unsigned nloops = nthreads / 512 + 1;
	nthreads = min(nthreads, 512);

	for (unsigned i = 0; i < nwavelengths; i++) {
		/*
		 * prepare right hand side
		 *
		 * nz: index of zero element. let's run with norders threads when
		 * possible. as this most usually will be below 512, let's just run one
		 * block.
		 *
		 * this could be done in parallel for every wavelength!
		 */
		const dim3 blocks(1);
		const dim3 threads(nthreads);

		prep[pol]<<<blocks, threads>>>(-order.lo, l, dev_mem->full_A,
				dev_mem->full_B, dev_kt, dev_kr, nloops, i,
				norders, nwavelengths, coeffs,
				nslices * nwavelengths, nslices);
		cudaThreadSynchronize();
		CUDA_CHECK(prep[pol], 2);


		for (unsigned j = 0; j < nslices; j++) {
			cuda_secular(
				pol,
				norders,
				coeffs,
				j + i * nslices,		// eps_m
				nwavelengths * nslices,		// eps rowcount
				dev_q,
				i,
				nwavelengths,
				slices[j].d,
				dev_mem);
		}

		// solve. write result somewhere into global memory. if memory
		// is not sufficiently large enough, emit right now, or grab
		// from memory and store to host memory
	}
}
