#ifndef __WAVEVEC_H__20427F61_52E1_4B09_A987_59B5EECC81D1
#define __WAVEVEC_H__20427F61_52E1_4B09_A987_59B5EECC81D1

#include "cutypes.h"
#include "types.hpp"

extern "C" void
cuda_wavevec(unsigned nslices, slice_entry_t *dev_slices,
	cuFloatComplex *dev_refrac_idx, unsigned nwavelengths,
	unsigned nmats,
	float *dev_wavelength,  order_t order, const float T,
	step_entry_t *steps, float theta, float *dev_q,
	cuFloatComplex *dev_kt, cuFloatComplex *dev_kr);

#endif /* __WAVEVEC_H__20427F61_52E1_4B09_A987_59B5EECC81D1 */

