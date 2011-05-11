#ifndef __SECULAR_H__14DC9079_F519_4525_A6C5_9DBF96D966BE
#define __SECULAR_H__14DC9079_F519_4525_A6C5_9DBF96D966BE

#include "types.hpp"
#include "cutypes.h"

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
		dev_mem_ptr_t *dev_mem);


#endif /* __SECULAR_H__14DC9079_F519_4525_A6C5_9DBF96D966BE */

