#ifndef __INVEPS_H__EBA18650_53F1_4B59_9BDF_BC84C1F156A4
#define __INVEPS_H__EBA18650_53F1_4B59_9BDF_BC84C1F156A4

#include <cuComplex.h>
#include "cutypes.h"

/*
 * calculate the inverse of the 'eps-matrix' of row m.
 *
 * eps:		matrix containing all eps values
 * nows:	number of rows in eps
 * m:		row of eps values to build eps-matrix from
 * n:		number of elements in one eps row
 * piv:		pivot element number of eps (usually n>>1)
 * epsinv:	output storage pointer. has to be of size NxN
 *
 */
extern "C"
void cuda_inveps(cuFloatComplex *eps, unsigned nrows, unsigned m ,
		unsigned n, cuFloatComplex *epsinv, dev_mem_ptr_t *dev_mem);


#endif /* __INVEPS_H__EBA18650_53F1_4B59_9BDF_BC84C1F156A4 */

