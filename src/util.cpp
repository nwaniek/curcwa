#include "util.hpp"
#include <iostream>
#include "cmplxutil.hpp"

// stupid interpolation implementation. could be done better, but maybe later...
cuFloatComplex
linterpol(const material *m, const float wl)
{
	if (m->nrefs == 1)
		return m->refs[0].refraction;

	unsigned i = 0;
	for (; (i < m->nrefs) && (wl > m->refs[i].wavelength); ++i);
	if (i == 0)
		return m->refs[0].refraction;

	// std::cout << i;

	float xp = m->refs[i].wavelength;
	float xm = m->refs[i-1].wavelength;
	cuFloatComplex yp = m->refs[i].refraction;
	cuFloatComplex ym = m->refs[i-1].refraction;

	cuFloatComplex result =  ym + (yp - ym) / (xp - xm) * (wl - xm);
	return result;
}

