#ifndef __TYPES_HPP__4161B3E3_5C0D_47F2_8B36_7CDEB0F215A7
#define __TYPES_HPP__4161B3E3_5C0D_47F2_8B36_7CDEB0F215A7

#include <cublas.h>

/**
 *
 */
typedef struct { int lo; int hi; } order_t;


/**
 * enum rcwa_approach - enumeration of different available rcwa calculation
 * approaches available
 */
typedef enum rcwa_approach
{
	RA_PARTIAL = 0,
	RA_FULL,

	RA_COUNT
} rcwa_approach_t;


/**
 * enum for polarization
 */
typedef enum {
	POL_TE,
	POL_TM,

	POL_COUNT
} pol_t;


/**
 * struct refractive_index - store the refractive index for a wavelength
 *
 * wavelength:	the wavelength the refractive index is bound to
 * refraction:	the refraction at the wavelength
 */
struct refractive_index
{
	float wavelength;
	cuFloatComplex refraction;
};


/**
 * struct material - material definition by refractive indices
 *
 * nrefs:	number of refractive indices pointer to by refs
 * refs:	simple array of all refractive indices
 */
struct material
{
	unsigned nrefs;
	const refractive_index *refs;
};


/**
 * struct step - formulate one step of a slice
 *
 * the material index has to be the index of the material returned by the
 * setup-function "material". so, when material is 1, it is expected to be the
 * second material returned by get_materials
 *
 * x:		x position, means: step starts from the previos step until x
 * mat_idx:	material the step is made of
 */
struct step
{
	float x;
	unsigned mat_idx;
};


/**
 * struct slice - a piece of the grating
 *
 * d:		thickness of the slice
 * nsteps:	number of steps within this slice
 * step:	array of nsteps steps
 */
struct slice
{
	float d;
	unsigned nsteps;
	const struct step *step;
};


/**
 * struct stack - the complete grating stack
 *
 * A Stack is made up of nslices count slices of material
 *
 * nslices:	number of slices within this stack
 * slice:	array with all slices the stack is made of
 */
struct stack
{
	unsigned nslices;
	const struct slice *slice;
};


#endif /* __TYPES_HPP__4161B3E3_5C0D_47F2_8B36_7CDEB0F215A7 */

