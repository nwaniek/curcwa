#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#define CULA_USE_CUDA_COMPLEX
#include <cula.hpp>
#include "setup.hpp"
#include "util.hpp"
#include "cmplxutil.hpp"
#include "cutypes.h"
#include "cu/fouriercoeff.h"
#include "cu/wavevec.h"
#include "cu/full.h"
#include "cu/partialt.h"
#include "cu/partialr.h"

#include <sys/time.h>
#include <unistd.h>
#include <cstdio>


/*
 * namespace encapsulation for global memory. this will contain scalar values
 * that are used for both host and device. may consider to remove the namespace
 * alltogether and use the variables without namespace
 */
namespace global {
	//! number of different materials used
	static unsigned nmats = 0;

	//! number of wavelengths to calculate RCWA for
	static unsigned nwavelengths = 0;

	//! incident angle of ray hitting the diffraction grating
	static float theta = 0.0f;

	//! diffraction orders to calculate RCWA for
	static order_t order = {0, 0};

	//! total number of orders
	static unsigned norders = 0;

	//! slice length
	static float T = 0.0f;

	// number of clices in the stack
	static unsigned nslices = 0;

	// number of overall-steps: sum(step * slices per step)
	static unsigned nsteps = 0;

	// polarization. this determines which calculation will be done: either
	// for TM or TE polarized waves
	static pol_t pol;
} // global


/*
 * namespace encapsulation for host memory
 */
namespace host {
	// all wavelengths
	static float *wavelength = NULL;

	//! pointer to the stack as defined by setup.cpp
	static const struct stack* stack;

	//! contains all refractive indices for every wavelength
	static cuFloatComplex *refrac_idx = NULL;

	// map containing two entries per step: number of steps, offset into
	// step-array. size: 2 * nslices
	static slice_entry_t *slice = NULL;

	// entries for all steps
	static step_entry_t *steps = NULL;
} // host


/*
 * namespace encapsulation for device memory
 * TODO: move those pointers into the dev_mem_ptr_t to reduce the number of
 * required function arguments to cuda-extern-calls
 */
namespace dev {
	static float *wavelength = NULL;
	static cuFloatComplex *refrac_idx = NULL;

	static slice_entry_t *slice = NULL;
	static step_entry_t *steps = NULL;

	// memory buffer for all fourier coefficiencs. This will be an MxN
	// matrix with nslices*nwavelengths rows and
	// ordercount * 2 + 1 cols.
	//
	// note: slices are grouped together, so, say you have 5 wavelengths and
	// 3 slices, the first 3 rows will be for the first wavelength, the next
	// 3 rows for the next wavelength, etc.
	static cuFloatComplex *fouriercoeffs = NULL;

	// matrix of size MxN, where M = nwavelengths and N = order.hi -
	// order.lo + 1. stores all q-values
	//
	// NOTE: q might be stored either COL or ROW major. this is switchable
	// by setting the appropriate define in global.h
	static float *q = NULL; // note: q will be stored row-major!!!

	// matrix of size MxN, where M = nwavelengths and N = order.hi -
	// order.lo + 1. stores all kt values
	static cuFloatComplex *kt = NULL;

	// matrix of size MxN, where M = nwavelengths and N = order.hi -
	// order.lo + 1. stores all kr values
	static cuFloatComplex *kr = NULL;

	// vector for the system Ax = b that will be solved.
	// requires memory for 2 * n * (l+1) elements, where n = #orders, l =
	// #slices - 2 (incident and transmission region removed); times
	// wavelength. -> MxN matrix with M = #wavelengths, N = 2*n*(l+1)
	// static cuFloatComplex *b = NULL;
	// is now in dev_mem_ptr_t -> full_B


	// dev mem pointer -> stores memory associated to calculation steps.
	// mostly temporary memory, but allocation takes much time.
	static dev_mem_ptr_t dev_mem = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
} // dev



void
release_dev_mem_ptr(dev_mem_ptr_t *dev_mem)
{
	cublasFree(dev_mem->full_A);
	cublasFree(dev_mem->full_B);
	cublasFree(dev_mem->full_lambda);
	cublasFree(dev_mem->full_v);
	cublasFree(dev_mem->full_vl);
	cublasFree(dev_mem->full_vx);
	cublasFree(dev_mem->full_vlx);

	cublasFree(dev_mem->secular_A);
	cublasFree(dev_mem->secular_C);
	cublasFree(dev_mem->secular_epsmat);
	cublasFree(dev_mem->secular_epsinv);

	cublasFree(dev_mem->inveps_A);
	cudaFree(dev_mem->inveps_ipiv);
}



static void
alloc_dev_mem()
{
	using namespace dev;

	if (!wavelength)
		CUDA_GUARD(cudaMalloc, (void**)&dev::wavelength, sizeof(float) *
				global::nwavelengths);

	if (!refrac_idx)
		CUDA_GUARD(cudaMalloc, (void**)&dev::refrac_idx, sizeof(cuFloatComplex) * global::nmats * global::nwavelengths);

	// allocate memory for full computation. TODO: make this dependent on
	// the approach
	if (!dev_mem.full_A) {
		const unsigned n = global::norders;
		const unsigned l = global::nslices - 2;
		const unsigned nn = 2 * n * (l + 1);
		// const unsigned nelems = global::nwavelengths * nn;

		// full
		CUBLAS_GUARD(cublasAlloc, nn * nn, sizeof(cuFloatComplex), (void**)&dev_mem.full_A);
		CUBLAS_GUARD(cublasAlloc, nn, sizeof(cuFloatComplex), (void**)&dev_mem.full_B);
		CUBLAS_GUARD(cublasAlloc, n, sizeof(cuFloatComplex), (void**)&dev_mem.full_lambda);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.full_v);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.full_vl);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.full_vlx);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.full_vx);

		// secular
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.secular_A);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.secular_C);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.secular_epsmat);
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.secular_epsinv);

		// inveps
		CUBLAS_GUARD(cublasAlloc, n * n, sizeof(cuFloatComplex), (void**)&dev_mem.inveps_A);
		CUDA_GUARD(cudaMalloc, (void**)&dev_mem.inveps_ipiv, sizeof(culaDeviceInt) * n);
	}
}


static void
copy_to_dev()
{
	CUDA_GUARD(cudaMemcpy, dev::wavelength, host::wavelength,
			sizeof(float) * global::nwavelengths,
			cudaMemcpyHostToDevice);

	CUDA_GUARD(cudaMemcpy, dev::refrac_idx, host::refrac_idx,
			sizeof(cuFloatComplex) * global::nmats * global::nwavelengths,
			cudaMemcpyHostToDevice);
}


static void
dump_refrac_matrix()
{
	std::cout << "Refrac Matrix: " << std::endl;

	for (unsigned i = 0; i < global::nmats; i++) {
		for (unsigned j = 0; j < global::nwavelengths; j++)
			std::cout << host::refrac_idx[IDX(i, j, global::nmats)] << " ";
		std::cout << std::endl;
	}

	std::cout << std::endl;
}


static void
dump_wavevec()
{
	std::cout << "Wave Vectors: " << std::endl;


	std::cout << std::endl;
}


static void
dump_fouriercoeffs()
{
	std::cout << "Fourier Coefficients: " << std::endl;

	// CUDA_GUARD(cudaMallocHost, (void**)&host::refrac_idx, sizeof(cuFloatComplex) * global::nmats * global::nwavelengths);

	cuFloatComplex *h;
	size_t nelems = global::nslices * global::nwavelengths * (2 * global::norders + 1);

	h = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * nelems);
	if (!h)
		std::cout << "h could not be allocated!" << std::endl;

	std::cout << nelems << std::endl;
	size_t s = nelems * sizeof(cuFloatComplex);
	std::cout << "nelems: " << nelems << " size: " << s << std::endl;
	CUDA_GUARD(cudaMemcpy, h, dev::fouriercoeffs, s, cudaMemcpyDeviceToHost);

	unsigned rowcount = global::nslices * global::nwavelengths;
	for (unsigned i = 0; i < rowcount; i++) {
		for (unsigned j = 0; j < (2 * global::norders + 1); j++) {
			std::cout << h[IDX(i, j, rowcount)] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	free(h);
}


static void
prepare_material(const material *m, unsigned i)
{
	for (unsigned j = 0; j < global::nwavelengths; j++) {
		// interpolate
		host::refrac_idx[IDX(i, j, global::nmats)] = linterpol(m, host::wavelength[j]);
		// store squared
		csquarei(host::refrac_idx[IDX(i, j, global::nmats)]);
	}
}


static void
prepare_materials()
{
	const material **m = new const material*[global::nmats];
	if (!host::refrac_idx)
		CUDA_GUARD(cudaMallocHost, (void**)&host::refrac_idx, sizeof(cuFloatComplex) * global::nmats * global::nwavelengths);

	get_materials(m);
	for (unsigned i = 0; i < global::nmats; i++)
		prepare_material(m[i], i);

	delete[] m;
}


static void
prepare_wavelengths()
{
	using namespace host;

	float from, to, step;
	get_wavelengths(&from, &to, &step);
	global::nwavelengths = unsigned((to - from) / step);
	if (!wavelength)
		CUDA_GUARD(cudaMallocHost, (void**)&wavelength, sizeof(float) *
				global::nwavelengths);

	// #pragma omp parallel for
	for (unsigned i = 0; i < global::nwavelengths; i++)
		wavelength[i] = from + i * step;
}


static void
prepare_stack()
{
	// get information from setup -> shovel every required data to the GPU
	host::stack = get_stack();
	global::nslices = host::stack->nslices;
	if (!global::nslices) {
		std::cout << "ERROR: stack has no slices." << std::endl;
		exit(EXIT_FAILURE);
	}

	// build the memory offset map: calculate how many entries there will be
	// required overall
	global::nsteps = 0;
	for (unsigned i = 0; i < global::nslices; i++)
		global::nsteps += host::stack->slice[i].nsteps;

	// allocate memory for all step information
	if (!host::steps)
		CUDA_GUARD(cudaMallocHost, (void**)&host::steps, sizeof(step_entry_t)
				* global::nsteps);
	if (!dev::steps)
		CUDA_GUARD(cudaMalloc, (void**)&dev::steps, sizeof(step_entry_t)
				* global::nsteps);
	// allocate memory for the step map
	if (!host::slice)
		CUDA_GUARD(cudaMallocHost, (void**)&host::slice,
				sizeof(slice_entry_t) * global::nslices);
	if (!dev::slice)
	CUDA_GUARD(cudaMalloc, (void**)&dev::slice, sizeof(slice_entry_t)
			* global::nslices);

	// fill the map and the step-array
	unsigned offset = 0;
	// #pragma omp parallel for
	for (unsigned i = 0; i < global::nslices; i++) {
		host::slice[i].offset = offset;
		host::slice[i].nsteps = host::stack->slice[i].nsteps;
		host::slice[i].d = host::stack->slice[i].d;

		for (unsigned j = 0; j < host::stack->slice[i].nsteps; j++, offset++) {
			host::steps[offset].x = host::stack->slice[i].step[i].x;
			host::steps[offset].m = host::stack->slice[i].step[i].mat_idx;
		}
	}

	// copy to device
	CUDA_GUARD(cudaMemcpy, dev::slice, host::slice,
			sizeof(slice_entry_t) * global::nslices,
			cudaMemcpyHostToDevice);
	CUDA_GUARD(cudaMemcpy, dev::steps, host::steps, sizeof(step_entry_t)
			* global::nsteps, cudaMemcpyHostToDevice);
}


static void
prepare_slices()
{
	// T: slice length. this can be easily determined by taking the last slice's
	//    last step x value; make sure that each slice has the same length

	slice sl = host::stack->slice[0];
	global::T = sl.step[sl.nsteps -1].x;
	for (unsigned i = 1; i < host::stack->nslices; i++) {
		sl = host::stack->slice[i];
		if (global::T != sl.step[sl.nsteps - 1].x) {
			std::cout << "ERROR: slices have unequal slice length" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}


static void
gc()
{
	// computation memory
	release_dev_mem_ptr(&dev::dev_mem);

	// fourier coeff memory
	cublasFree(dev::kr);
	cublasFree(dev::kt);
	cublasFree(dev::q);
	cublasFree(dev::fouriercoeffs);

	// slice stuff memory
	cudaFree(dev::steps);
	cudaFree(dev::slice);
	cudaFreeHost(host::steps);
	cudaFreeHost(host::slice);

	// refrac memory
	cudaFree(dev::refrac_idx);
	cudaFree(dev::wavelength);
	cudaFreeHost(host::refrac_idx);
	cudaFreeHost(host::wavelength);

	culaShutdown();
	cublasShutdown();
}


static void
rcwa_fourier()
{
	// allocate enough memory to store every fourier coeff if the memory is
	// not already allocated
	if (!dev::fouriercoeffs) {
		CUBLAS_GUARD(cublasAlloc, global::nslices * (2 * global::norders + 1) *
				global::nwavelengths, sizeof(cuFloatComplex),
				(void**)&(dev::fouriercoeffs));

		CUDA_GUARD(cudaMemset, dev::fouriercoeffs, 0,
				sizeof(cuFloatComplex) * global::nslices * global::nwavelengths * (2 * global::norders + 1));
	}

	cuda_fouriercoeff(global::nwavelengths, dev::refrac_idx,
			global::nslices, dev::slice, dev::steps, global::norders,
			dev::fouriercoeffs, global::nmats);
}


static void
rcwa_wavevec()
{
	// allocate space for the wave-vectors if not already done
	if (!dev::q) {
		size_t n = global::norders * global::nwavelengths;

		CUBLAS_GUARD(cublasAlloc, n, sizeof(float), (void**)&dev::q);
		CUBLAS_GUARD(cublasAlloc, n, sizeof(cuFloatComplex),
				(void**)&dev::kt);
		CUBLAS_GUARD(cublasAlloc, n, sizeof(cuFloatComplex),
				(void**)&dev::kr);
	}

	// calculate all wave vectors for all wavelengths simultaneoously
	cuda_wavevec(global::nslices, dev::slice, dev::refrac_idx,
			global::nwavelengths, global::nmats, dev::wavelength,
			global::order, global::T, dev::steps,
			global::theta * M_PI / 180.0, // pass in theta in rad
			dev::q, dev::kt, dev::kr);
}


static void
rcwa_full()
{
	cuda_full(global::pol,
		  global::order,
		  global::nwavelengths,
		  global::nslices,
		  dev::kt,
		  dev::kr,
		  dev::q,
		  host::slice,
		  global::nmats,
		  dev::fouriercoeffs,
		  &dev::dev_mem
		 );
}


static void
rcwa_partialr()
{
	cuda_partialr();
}


static void
rcwa_partialt()
{
	cuda_partialt();
}


typedef struct _benchmark {
	double prepare_wavelengths;
	double prepare_materials;
	double prepare_stack;
	double prepare_slices;
	double alloc_dev_mem;
	double copy_to_dev;
	double fourier;
	double wavevec;
	double full;
	double partial_t;
	double partial_r;
} benchmark_t;


double delta_time(void (*f)(), bool do_bench = false)
{
	double result = 0.;
	if (do_bench) {
		struct timeval start, end;
		gettimeofday(&start, NULL);
		f();
		gettimeofday(&end, NULL);
		result = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
	}
	else
		f();
	return result;
}


int
main(int argc, const char **argv)
{
	benchmark_t benchtimes = {.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0};

	bool do_benchmark = false;
	unsigned nloops = 1;
	rcwa_approach_t approach = get_method();

	if (argc > 1) {
		do_benchmark = true;
		if (atoi(argv[1]) <= 0)
			return EXIT_SUCCESS;

		nloops = atoi(argv[1]);
		std::cout << "Benchmark run: " << nloops << " times. Mode:";
		switch (approach) {
		case RA_FULL:
			std::cout << " full.";
			break;
		case RA_PARTIAL:
			std::cout << " partial.";
			break;
		default:
			std::cerr << "ERROR: invalid rcwa approach selected" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (nloops == 1)
			std::cout << " Results will contain memory allocation time.";
		else
			std::cout << " Results will only contain computation time.";
		std::cout << std::endl;
	}

	atexit(gc);
	CUBLAS_GUARD(cublasInit);
	CULA_GUARD(culaInitialize);

	// read the setup for which to calculate the RCWA
	global::nmats = get_material_count();
	global::order = get_orders();
	global::norders = global::order.hi - global::order.lo + 1;
	global::theta = get_incident_angle();


#define BENCHMARK(f, x) { \
	double t = delta_time((f), do_benchmark); \
	if (i || nloops == 1) \
		(x) += t; }

	for (unsigned i = 0; i < nloops; i++) {
		BENCHMARK(prepare_wavelengths, benchtimes.prepare_wavelengths);
		BENCHMARK(prepare_materials, benchtimes.prepare_materials);
		BENCHMARK(prepare_stack, benchtimes.prepare_stack);
		BENCHMARK(prepare_slices, benchtimes.prepare_slices);

		// get memory buffers that have yet to be allocated and copy data to the
		// device
		BENCHMARK(alloc_dev_mem, benchtimes.alloc_dev_mem);
		BENCHMARK(copy_to_dev, benchtimes.copy_to_dev);

		// debug stuff
		if (do_benchmark && nloops == 1 && argc > 2)
			dump_refrac_matrix();

		//
		// RCWA : the real algorithmic work is done within the following functions
		//
		// 1. calculate all fourier coefficients
		BENCHMARK(rcwa_fourier, benchtimes.fourier);
		cudaThreadSynchronize();
		if (do_benchmark && nloops == 1 && argc > 2)
			dump_fouriercoeffs();

		// 2. calculate all wave vectors, meaning, where the waves will be
		// propagated to
		BENCHMARK(rcwa_wavevec, benchtimes.wavevec);
		if (do_benchmark && nloops == 1 && argc > 2)
			dump_wavevec();

		// 3. select which method to use. there's either the FULL mode or the
		// PARTIAL mode. In the latter, reflectance and transmittance will be
		// calculated separately
		switch (approach) {
		case RA_FULL:
			BENCHMARK(rcwa_full, benchtimes.full);
			break;
		case RA_PARTIAL:
			BENCHMARK(rcwa_partialr, benchtimes.partial_r);
			BENCHMARK(rcwa_partialt, benchtimes.partial_t);
			break;
		default:
			std::cout << "ERROR: invalid rcwa approach selected" << std::endl;
			exit(EXIT_FAILURE);
		}

		cudaThreadSynchronize();
		// 4. done: output the results
		//

		if (do_benchmark)
			std::cout << "." << std::flush;
	}
#undef BENCHMARK

	if (do_benchmark) {
		if (nloops > 1) {
			benchtimes.alloc_dev_mem = 0.;
			nloops--;
		}

		std::cout << std::endl;
		benchtimes.prepare_wavelengths /= nloops;
		benchtimes.prepare_materials /= nloops;
		benchtimes.prepare_stack /= nloops;
		benchtimes.prepare_slices /= nloops;
		benchtimes.alloc_dev_mem /= nloops;
		benchtimes.copy_to_dev /= nloops;
		benchtimes.fourier /= nloops;
		benchtimes.wavevec /= nloops;
		benchtimes.full /= nloops;
		benchtimes.partial_t /= nloops;
		benchtimes.partial_r /= nloops;

		double sum = benchtimes.prepare_wavelengths +
			benchtimes.prepare_materials +
			benchtimes.prepare_stack +
			benchtimes.prepare_slices +
			benchtimes.alloc_dev_mem +
			benchtimes.copy_to_dev +
			benchtimes.fourier +
			benchtimes.wavevec +
			benchtimes.full +
			benchtimes.partial_t +
			benchtimes.partial_r;

		std::cout <<
			"prepare_wavelengths: " << benchtimes.prepare_wavelengths << " ms" << std::endl <<
			"prepare_materials:   " << benchtimes.prepare_materials << " ms" << std::endl <<
			"prepare_stack:       " << benchtimes.prepare_stack << " ms" << std::endl <<
			"prepare_slices:      " << benchtimes.prepare_slices << " ms" << std::endl <<
			"alloc_dev_mem:       " << benchtimes.alloc_dev_mem << " ms" << std::endl <<
			"copy_to_dev:         " << benchtimes.copy_to_dev << " ms" << std::endl <<
			"fourier:             " << benchtimes.fourier << " ms" << std::endl <<
			"wavevec:             " << benchtimes.wavevec << " ms" << std::endl <<
			"full:                " << benchtimes.full << " ms" << std::endl <<
			"partial_t:           " << benchtimes.partial_t << " ms" << std::endl <<
			"partial_r:           " << benchtimes.partial_r << " ms" << std::endl <<
			"sum:                 " << sum << " ms" << std::endl;
	}

	return EXIT_SUCCESS;
}
