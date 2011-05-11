#include "cu/inveps.h"
#include <cublas.h>
#define CULA_USE_CUDA_COMPLEX
#include <cula.h>
#include "util.hpp"
#include "cmplxutil.hpp"


__global__ void
d_epsinvdiag(cuFloatComplex *epsinv, unsigned n)
{
	// assume there are enough blocks that every block has to do work just
	// once.
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	epsinv[IDX(i, i, n)].x = 1.f;
}


__global__ void
d_epsdiag(cuFloatComplex *eps, unsigned nrows, unsigned m, unsigned n,
		cuFloatComplex *A)
{
	// assume there are enough blocks that every block has to just do work
	// once.
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	// walk over every column
	for (unsigned j = 0; j < n; j++)
		A[IDX(i, j, n)] = eps[IDX(m, i - j + n, nrows)];
}


/*
 * runs about 1ms faster than the other variant
 */
__global__ void
d_epsdiag_shm(cuFloatComplex *eps, unsigned nrows, unsigned m, unsigned n,
		cuFloatComplex *A)
{
	extern __shared__ cuFloatComplex shared[];

	// assume there are enough blocks that every block has to just do work
	// once.
	const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	// load all into shared memory
	unsigned k = 0;
	for (; k < (2 * n + 1); k += blockDim.x)
		shared[k] = eps[IDX(m, k, nrows)];
	__syncthreads();

	// walk over every column
	for (unsigned j = 0; j < n; j++)
		A[IDX(i, j, n)] = shared[i - j + n];
}



extern "C"
void cuda_inveps(cuFloatComplex *eps,   // matrix with all fourier coefficients
		unsigned nrows,         // number of rows in eps
		unsigned m,             // row index to use
		unsigned n,             // number of orders =:= size of NxN matrix
		cuFloatComplex *epsinv, // output: inverted matrix
		dev_mem_ptr_t *dev_mem  // pointers to device memory
		)
{
	dim3 nblocks((n / 32) + 1);
	dim3 nthreads(32);

	/*
	 * prepare A
	 */
	cuFloatComplex *A = dev_mem->inveps_A;
	CUDA_GUARD(cudaMemset, A, 0, n * n * sizeof(cuFloatComplex));
#if INVEPS_EPSDIAG_SHM
	size_t size = sizeof(cuFloatComplex) * (2 * n + 1);
	d_epsdiag_shm<<<nblocks, nthreads, size>>>(eps, nrows, m, n, A);
#else
	d_epsdiag<<<nblocks, nthreads>>>(eps, nrows, m, n, A);
#endif
	cudaThreadSynchronize();
	CUDA_CHECK(d_epsdiag, 2);

	/*
	 * prepare epsinv matrix
	 */
	CUDA_GUARD(cudaMemset, epsinv, 0, n * n * sizeof(cuFloatComplex));
	d_epsinvdiag<<<nblocks, nthreads>>>(epsinv, n);
	cudaThreadSynchronize();
	CUDA_CHECK(d_epsdiag, 2);

	/*
	 * invoke the cula solver
	 */

	/*
	 * debug output when things go horribly wrong
	 */
#if 0
	cuFloatComplex *x;
	x = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * n * n);
	cudaMemcpy(x, epsinv, sizeof(cuFloatComplex) * n * n, cudaMemcpyDeviceToHost);
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			std::cout << x[IDX(i, j, n)] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	cudaMemcpy(x, A, sizeof(cuFloatComplex) * n * n, cudaMemcpyDeviceToHost);
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			std::cout << x[IDX(i, j, n)] << " ";
		}
		std::cout << std::endl;
	}

	free(x);
	exit(EXIT_FAILURE);
#endif

	culaDeviceInt *ipiv = dev_mem->inveps_ipiv;
	// TODO: check cula info flag
	culaDeviceCgesv(n, n, A, n, ipiv, epsinv, n);
	//CULA_GUARD(culaDeviceCgesv, n, n, A, n, ipiv, epsinv, n);
}
