#ifndef __FULL_H__CB462B28_8590_49EB_A236_E3E733F2023A
#define __FULL_H__CB462B28_8590_49EB_A236_E3E733F2023A

#include "types.hpp"
#include "cutypes.h"

extern "C"
void cuda_full(pol_t pol, order_t order, unsigned nwavelengths,
		unsigned nslices,
		cuFloatComplex *dev_kt, cuFloatComplex *dev_kr,
		float *dev_q, slice_entry_t *slices, unsigned nmats,
		cuFloatComplex *coeffs, dev_mem_ptr_t *dev_mem
	      );

#endif /* __FULL_H__CB462B28_8590_49EB_A236_E3E733F2023A */

