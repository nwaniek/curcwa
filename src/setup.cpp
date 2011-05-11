#include "setup.hpp"
#include <cublas.h>

/*
 * enumerate all materials. though this is not required, it makes it more easy
 * to define the slices
 */
enum materials {
	MAT_AIR,
	MAT_GOLD,

	MAT_COUNT
};

/*
 * material setup for air: air has a refractive index of 1.0+0.0j for every
 * wavelength, therefore just one refractive index is required
 */
static const unsigned air_nrefs = 1;
static const refractive_index air_refs[air_nrefs] = {
	{ 1.0, {1.0, 0.0} }
};
static const material air = {air_nrefs, air_refs};


/*
 * gold has different refractive indices for different wavelenghts. you can
 * provide all refractive indices within an array of contained type
 * refractive_index.
 * it is expected that the materials wavelength indices are in ascending order.
 * not complying to this will break the interpolation
 */
static const unsigned gold_nrefs = 54;
static const refractive_index gold_refs[gold_nrefs] = {
	{ 0.2883, {1.74  , 1.900} },
	{ 0.2952, {1.776 , 1.918} },
	{ 0.3024, {1.812 , 1.920} },
	{ 0.3100, {1.830 , 1.916} },
	{ 0.3179, {1.840 , 1.904} },
	{ 0.3263, {1.824 , 1.878} },
	{ 0.3351, {1.798 , 1.960} },
	{ 0.3444, {1.766 , 1.846} },
	{ 0.3542, {1.740 , 1.848} },
	{ 0.3647, {1.716 , 1.862} },
	{ 0.3757, {1.696 , 1.906} },
	{ 0.3875, {1.674 , 1.936} },
	{ 0.4000, {1.658 , 1.956} },
	{ 0.4133, {1.636 , 1.958} },
	{ 0.4275, {1.616 , 1.940} },
	{ 0.4428, {1.562 , 1.904} },
	{ 0.4592, {1.426 , 1.846} },
	{ 0.4769, {1.242 , 1.796} },
	{ 0.4959, {0.916 ,  1.84} },
	{ 0.5166, {0.608 ,  2.12} },
	{ 0.5391, {0.402 ,  2.54} },
	{ 0.5636, {0.306 ,  2.88} },
	{ 0.5904, {0.180 ,  2.84} },
	{ 0.6199, {0.130 ,  3.16} },
	{ 0.6526, {0.166 ,  3.15} },
	{ 0.6888, {0.160 ,   3.8} },
	{ 0.7293, {0.164 ,  4.35} },
	{ 0.7749, {0.174 ,  4.86} },
	{ 0.8266, {0.188 ,  5.39} },
	{ 0.8856, {0.210 ,  5.88} },
	{ 0.9537, {0.236 ,  6.47} },
	{ 1.033 , {0.272 ,  7.07} },
	{ 1.127 , {0.312 ,  7.93} },
	{ 1.240 , {0.372 ,  8.77} },
	{ 1.265 , {0.389 ,  8.09} },
	{ 1.291 , {0.403 ,  8.25} },
	{ 1.319 , {0.419 ,  8.42} },
	{ 1.384 , {0.436 ,  8.59} },
	{ 1.378 , {0.454 ,  8.77} },
	{ 1.409 , {0.473 ,  8.96} },
	{ 1.442 , {0.493 ,  9.15} },
	{ 1.476 , {0.515 ,  9.36} },
	{ 1.512 , {0.537 ,  9.58} },
	{ 1.550 , {0.559 ,  9.81} },
	{ 1.590 , {0.583 ,  10.1} },
	{ 1.631 , {0.609 ,  10.3} },
	{ 1.675 , {0.636 ,  10.6} },
	{ 1.722 , {0.665 ,  10.9} },
	{ 1.771 , {0.696 ,  11.2} },
	{ 1.823 , {0.730 ,  11.5} },
	{ 1.879 , {0.767 ,  11.9} },
	{ 1.937 , {0.807 ,  12.2} },
	{ 2.000 , {0.850 ,  12.6} },
	{ 2.066 , {0.896 ,  13.0} }
};
static const material gold = {gold_nrefs, gold_refs};

/*
 * for the moment let's just make a simple binary gold grating. incident and
 * transmission are slices as well, so we have to model them.
 */

// setup for the binary grating slice
static const step binary_steps[] = {{0.5, MAT_AIR}, {1.0, MAT_GOLD}};
static const slice binary = {1.0, 2, binary_steps};

// setup for the transmission slice
static const step trans_steps[] = {{1.0, MAT_AIR}};
static const slice transmission = {1.0, 1, trans_steps};

// setup for the incident slice
static const step inc_steps[] = {{1.0, MAT_GOLD}};
static const slice incident = {1.0, 1, inc_steps};

// make one complete stack out of the individual slices
static const slice slices[] = {incident, binary, transmission};
static const struct stack stack = {3, slices};

// return the full stack when asked for
const struct stack*
get_stack()
{
	return &stack;
}

// calculate orders from -12 to 12
order_t
get_orders()
{
	static order_t o = {-48, 48};
	return o;
}

// the incident angle for which the calculation shall take place
float
get_incident_angle()
{
	return 60.0f;
}

// return which approach shall be used for computation.
rcwa_approach_t get_method()
{
	// either RA_FULL or RA_PARTIAL
	return RA_FULL;
}

// the backend needs to know how many materials there will be for allocating
// memory
unsigned
get_material_count()
{
	return MAT_COUNT;
}

// return an array with all materials that will be used
void get_materials(const material **m) {
	m[MAT_AIR]  = &air;
	m[MAT_GOLD] = &gold;
}


// tell the backend which wavelengths to calculate
void
get_wavelengths(float *from, float *to, float *step)
{
	*from = .5;
	*to = 1.2;
	*step = 0.01;
}

// which polarization shall be computed. can be either TM or TE
pol_t
get_polarization()
{
	return POL_TM;
}
