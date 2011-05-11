#ifndef __FOURIERCOEFF_H__148F1B72_C58B_4792_AD28_83D3AE630796
#define __FOURIERCOEFF_H__148F1B72_C58B_4792_AD28_83D3AE630796

#include "cuComplex.h"
#include "cutypes.h"

extern "C"
void cuda_fouriercoeff(unsigned n_wavelengths, cuFloatComplex *dev_refrac_idx,
	unsigned nslices, slice_entry_t *dev_slices,
	step_entry_t *dev_steps, int norders, cuFloatComplex *fouriercoeffs,
	unsigned nmats);


#endif /* __FOURIERCOEF_H__148F1B72_C58B_4792_AD28_83D3AE630796 */

