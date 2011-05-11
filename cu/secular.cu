#include "cu/secular.h"
#include "clapack.h"
#include <cublas.h>
#define CULA_USE_CUDA_COMPLEX
#include <cula.h>
#include "cmplxutil.hpp"
#include "cu/inveps.h"
#include "util.hpp"
#include "global.h"


/*
 *
 * functions that share the same work accross both TM and TE specialization
 *
 */

__global__ void
d_sqrt(cuFloatComplex *x, const unsigned n)
{
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	cuFloatComplex tmp = x[i];
	csqrti(tmp);
	x[i] = tmp;
}


__global__ void
d_compute_result(cuFloatComplex *lambda, cuFloatComplex *v, cuFloatComplex *vl,
		cuFloatComplex *vx, cuFloatComplex *vlx, const unsigned n,
		const float d)
{
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	cuFloatComplex l = lambda[i];
	csqrti(l);
	cuFloatComplex z = l * d * -1.f;
	cexpi(z);

	for (unsigned j = 0; j < n; j++) {
		__syncthreads();
		cuFloatComplex _v = v[IDX(j, i, n)];
		cuFloatComplex _vl = l * _v;

		vl[IDX(j, i, n)] = _vl;
		vx[IDX(j, i, n)] = _v * z;
		vlx[IDX(j, i, n)] = _vl * z;
	}

	lambda[i] = l;
}



/*
 * runs approximately 8ms faster than the non-shared-memory version. uses shared
 * memory to coalesc global writes
 */
__global__ void
d_compute_result_shm(cuFloatComplex *lambda, cuFloatComplex *v, cuFloatComplex *vl,
		cuFloatComplex *vx, cuFloatComplex *vlx, const unsigned n, // n = norders
		const float d)
{
	// memory for l and z
	extern __shared__ cuFloatComplex shared[];

	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	cuFloatComplex l, z;
	unsigned k = threadIdx.x;
	for (; k < n; k += blockDim.x) {
		l = lambda[k];
		csqrti(l);
		z = l * d * -1.f;
		cexpi(z);

		shared[k] = l;
		shared[k + n] = z;
	}
	__syncthreads();
	lambda[i] = shared[i];

	for (unsigned j = 0; j < n; j++) {
		z = shared[n + j];
		l = shared[j];

		cuFloatComplex _v = v[IDX(i, j, n)];
		cuFloatComplex _vl = l * _v;

		vl[IDX(i, j, n)] = _vl;
		vx[IDX(i, j, n)] = _v * z;
		vlx[IDX(i, j, n)] = _vl * z;
	}
}



/*
 *
 * TM polarization specific functions
 *
 */



/*
 * n: number of diffraction orders and therefore sizoe of A: NxN
 */
__global__ void
d_prepare_evm_tm(const unsigned n, cuFloatComplex *A, const unsigned eps_m,
		cuFloatComplex *eps, unsigned nrows)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	for (unsigned j = 0; j < n; j++)
		A[IDX(i, j, n)] = eps[IDX(eps_m, i - j + n, nrows)];
}


/*
 * approximately one millisecond faster than regular prepare_evm_tm
 */
__global__ void
d_prepare_evm_tm_shm(const unsigned n, cuFloatComplex *A, const unsigned eps_m,
		cuFloatComplex *eps, unsigned nrows)
{
	extern __shared__ cuFloatComplex shared[];

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	// load all eps values into shared memory. ugly: scattered global memory read
	unsigned k = threadIdx.x;
	for (; k < n; k += blockDim.x) {
		shared[k] = eps[IDX(eps_m, k, nrows)];
	}
	__syncthreads();

	for (unsigned j = 0; j < n; j++)
		A[IDX(i, j, n)] = shared[i - j + n]; // eps[IDX(eps_m, i - j + n, nrows)];
}



__global__ void
d_prepare_q_tm(const unsigned n, cuFloatComplex *A, float *q, const unsigned
		q_m, const unsigned nq)
{
	extern __shared__ float qshared[];
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	// load all q values into shared memory
	unsigned k = threadIdx.x;
	for (; k < n; k += blockDim.x)
#if ROW_MAJOR
		qshared[k] = q[IDXR(q_m, k, nq)];
#else
		qshared[k] = q[IDX(q_m, k, nq)];
#endif
	__syncthreads();

	for (unsigned j = 0; j < n; j++) {
		const cuFloatComplex tmp = A[IDX(i, j, n)];
		const float _q = qshared[i] * qshared[j];
		A[IDX(i, j, n)] = tmp * _q;

	}
}


void
secular_tm(const unsigned norders,
		cuFloatComplex *eps,   // fourier coefficients
		const unsigned eps_m,  // row index for eps
		const unsigned nrows,  // number of rows in eps
		float *q,              // wave vectors
		const unsigned q_m,    // row of wave vector matrix to use
		const unsigned nq,     // number of rows in q (if q is stored row major: number of cols)
		const float d,         // slice thickness
		dev_mem_ptr_t *dev_mem // device memory pointers
	  )
{
	static cuFloatComplex c_zero = {.0f, .0f};
	static cuFloatComplex c_one = {1.f, 0.f};


	cuFloatComplex *A = dev_mem->secular_A;
	cuFloatComplex *C = dev_mem->secular_C;
	cuFloatComplex *epsmat = dev_mem->secular_epsmat;
	cuFloatComplex *epsinv = dev_mem->secular_epsinv;

	/*
	 * prepare eigenvalue matrix
	 */
	dim3 blocks(norders / 32 + 1);
	dim3 threads(32);
	size_t shared;
#if SECULAR_PREPARE_EVM_SHM
	shared = (norders * 2 + 1) * sizeof(cuFloatComplex);
	d_prepare_evm_tm_shm<<<blocks, threads, shared>>>(norders, epsmat, eps_m, eps, nrows);
#else
	d_prepare_evm_tm<<<blocks, threads>>>(norders, epsmat, eps_m, eps, nrows);
#endif
	cudaThreadSynchronize();
	CUDA_CHECK(d_prepare_evm_tm, 2);

	/*
	 * prepare epsinv
	 */
	cuda_inveps(eps, nrows, eps_m, norders, epsinv, dev_mem);
	CUBLAS_GUARD_E(cublasCcopy, norders * norders, epsinv, 1, A, 1);

	/*
	 * prepare q matrix
	 */
	shared = sizeof(float) * norders;
	d_prepare_q_tm<<<blocks, threads, shared>>>(norders, A, q, q_m, nq);
	cudaThreadSynchronize();
	CUDA_CHECK(d_prepare_q_tm, 2);


	/*
	 * prepare for final solve
	 */
	CUDA_GUARD(cudaMemset, C, 0, sizeof(cuFloatComplex) * norders * norders);
	CUBLAS_GUARD_E(cublasCgemm, 'N', 'N', norders, norders, norders, c_one, epsmat, norders, A, norders,
				c_zero, C, norders);

	// FIXME: invoke CGEEV
	//
#if SECULAR_COMPUTE_RESULTS_SHM
	shared = sizeof(cuFloatComplex) * norders * 2;
	d_compute_result_shm<<<blocks, threads, shared>>>(dev_mem->full_lambda, dev_mem->full_v, dev_mem->full_vl, dev_mem->full_vx, dev_mem->full_vlx, norders, d);
#else
	d_compute_result<<<blocks, threads>>>(dev_mem->full_lambda, dev_mem->full_v, dev_mem->full_vl, dev_mem->full_vx, dev_mem->full_vlx, norders, d);
#endif
	cudaThreadSynchronize();
	CUDA_CHECK(d_compute_result, 2);

	// TODO: possible speedup: implement own matrix multiplication that
	// calculates vl and vlx at once, reducing memory reads of epsinv to
	// one-time instead of twice
	CUBLAS_GUARD_E(cublasCgemm, 'N', 'N', norders, norders, norders, c_one, epsinv, norders, dev_mem->full_vl, norders,
			c_zero, C, norders);



	CUDA_GUARD(cudaMemcpy, dev_mem->full_vl, C,
				sizeof(cuFloatComplex) * norders * norders,
				cudaMemcpyDeviceToDevice);

	CUBLAS_GUARD_E(cublasCgemm, 'N', 'N', norders, norders, norders, c_one, epsinv, norders, dev_mem->full_vlx, norders,
			c_zero, C, norders);
	CUDA_GUARD(cudaMemcpy, dev_mem->full_vlx, C,
				sizeof(cuFloatComplex) * norders * norders,
				cudaMemcpyDeviceToDevice);
}



/*
 *
 * TE polarization specific functions
 *
 */


/*
 * prepare the eigenvalue problem matrix A
 */
__global__ void
d_prepare_evm_te(const unsigned n, cuFloatComplex *A, const unsigned eps_m,
		cuFloatComplex *eps, const unsigned nrows, const unsigned piv)
{
	// assume there are enough blocks to run through the complete matrix A
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	for (unsigned j = 0; j < n; j++)
		A[IDX(i, j, n)] = eps[IDX(eps_m, i - j + piv, nrows)] * -1.f;
}


__global__ void
d_prepare_q_te(const unsigned n, cuFloatComplex *A, float *q, const unsigned
		q_m, const unsigned nq)
{
	// assume there are enough blocks running to span the hole matrix.
	unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n)
		return;

	float qj = q[IDX(q_m, j, nq)];
	qj *= qj;

	// as q is real valued, add to real-part of A
	// TODO: Check if this is proper math, a.k.a check if q really just has
	// to be real valued. have to check this throughout the whole code
	A[IDX(j, j, n)].x += qj;
}


void
secular_te(const unsigned n, cuFloatComplex *eps, const unsigned eps_m, const
		unsigned nrows, const unsigned piv, float *q, const unsigned
		q_m, const unsigned nq, const float d, dev_mem_ptr_t *dev_mem)
/*
, cuFloatComplex *v,
		cuFloatComplex *lambda, cuFloatComplex *vl, cuFloatComplex *vx,
		cuFloatComplex *vlx)
*/
{
	/*

	cuFloatComplex *A;
	CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&A);

	dim3 nblocks((n >> 5) + 1);
	dim3 nthreads(1 << 5);

	// prepare the eigenvalue problem matrix
	d_prepare_evm_te<<<nblocks, nthreads>>>(n, A, eps_m, eps, nrows, piv);
	d_prepare_q_te<<<nblocks, nthreads>>>(n, A, q, q_m, nq);

	// FIXME: missing: essential CGEEV call. will return values for lambda
	// and v
	// cgeev(...)

	// sqrt is moved into d_compute_result
	// d_sqrt<<<nblocks, nthreads>>>(lambda, n);

	// merge the computation of vl, vx and vlx into one function as
	// essentially the values depend partly on each other. therefore,
	d_compute_result<<<nblocks, nthreads>>>(lambda, v, vl, vx, vlx, n, d);

	// push the fucking shit back onto the device. yay.
	CUBLAS_GUARD(cublasFree, A);
	*/
}


/*
 * pol: job to do
 * norders: number of diffraction orders
 * eps: full eps matrix
 * eps_m: row of eps to calculate stuff for
 * nrows: numbers of rows in eps
 * piv: pivot element number of
 * q: full q matrix
 * q_m: row of q to use
 * nq: number of rows in q
 * d: thickness of slice
 * v: NxN array
 * lambda: array of length n
 * vl, vx, vlx: matrices of size NxN
 *
 * TODO: let the functions access d by slice-matrix, not by direct parameter!
 */
extern "C"
void cuda_secular(pol_t pol,
		const unsigned norders,
		cuFloatComplex *eps,
		unsigned eps_m,
		const unsigned nrows,
		float *q,
		const unsigned q_m,
		const unsigned nq,
		const float d,
		dev_mem_ptr_t *dev_mem)
{
	typedef void (*secular_f)(
			const unsigned n,
			cuFloatComplex *eps,
			const unsigned eps_m,
			const unsigned nrows,
			float *q,
			const unsigned q_m,
			const unsigned nq,
			const float d,
			dev_mem_ptr_t *dev_mem);

	/*
		, cuFloatComplex *v, cuFloatComplex *lambda,
			cuFloatComplex *vl, cuFloatComplex *vx, cuFloatComplex
			*vlx);
	*/

	static const secular_f secular[] = {/*secular_te,*/ secular_tm};
	secular[pol](norders, eps, eps_m, nrows, q, q_m, nq, d, dev_mem);
}
