#ifndef __CMPLXUTIL_HPP__FA512FD4_209C_4443_906F_610B0D132370
#define __CMPLXUTIL_HPP__FA512FD4_209C_4443_906F_610B0D132370


#include <ostream>


template <typename COMPLEX, typename REAL>
__device__ __host__ inline void cdivfi (COMPLEX &cmplx, REAL f)
{
	cmplx.x /= f;
	cmplx.y /= f;
}


template <typename COMPLEX, typename REAL>
__device__ __host__ inline void cmulfi (COMPLEX &cmplx, REAL f)
{
	cmplx.x *= f;
	cmplx.y *= f;
}


__device__ __host__ inline void cexpi(cuFloatComplex &c)
{
	float r = c.x;
	c.x = exp(r * cos(c.y));
	c.y = exp(r * sin(c.y));
}


template <typename COMPLEX, typename REAL>
__device__ __host__ inline COMPLEX operator* (const COMPLEX &c, const REAL f)
{
	COMPLEX result = {c.x * f, c.y * f};
	return result;
}


template <typename COMPLEX, typename REAL>
__device__ __host__ inline COMPLEX& operator*= (COMPLEX &c, const REAL f)
{
	c.x *= f;
	c.y *= f;
	return c;
}


template <typename COMPLEX>
__device__ __host__ inline COMPLEX operator/ (const COMPLEX &x, const COMPLEX &y)
{
	float frac = y.x * y.x + y.y * y.y;
	COMPLEX result = {(x.x * y.x + x.y * y*y) / frac, (x.y * y.x - x.x * y.y) / frac};
	return result;
}

// (a+ib)/(c+id)=(ac+bd+i(bc-ad))/(c2+d2)


template <typename COMPLEX, typename REAL>
__device__ __host__ inline COMPLEX operator/ (const COMPLEX &c, const REAL f)
{
	COMPLEX result = {c.x / f, c.y / f};
	return result;
}


template <typename COMPLEX, typename REAL>
__device__ __host__ inline COMPLEX& operator/= (COMPLEX &c, const REAL f)
{
	c.x /= f;
	c.y /= f;
	return c;
}



template <typename COMPLEX>
__device__ __host__ inline COMPLEX operator- (const COMPLEX &a, const COMPLEX &b)
{
	COMPLEX result = {a.x - b.x, a.y - b.y};
	return result;
}

template <typename COMPLEX>
__device__ __host__ inline COMPLEX& operator-= (COMPLEX &a, const COMPLEX &b)
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

template <typename COMPLEX>
__device__ __host__ inline COMPLEX operator+ (const COMPLEX &a, const COMPLEX &b)
{
	COMPLEX result = {a.x + b.x, a.y + b.y};
	return result;
}

template <typename COMPLEX>
__device__ __host__ inline COMPLEX& operator+= (COMPLEX &a, const COMPLEX &b)
{
	a.x += b.x;
	a.y += b.y;
	return a;
}


template <typename COMPLEX>
__device__ __host__ inline COMPLEX operator* (const COMPLEX &a, const COMPLEX &b)
{
	COMPLEX c = a;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}


__device__ __host__ inline cuFloatComplex& operator*= (cuFloatComplex &a, const cuFloatComplex &b)
{
	float r = a.x;
	float i = a.y;

	a.x = r * b.x - i * b.y;
	a.y = r * b.y + i * b.x;

	return a;
}


inline void
csquarei(cuFloatComplex &c)
{
	float r = c.x;
	float i = c.y;

	c.x = r * r - i * i;
	c.y = r * i + i * r;
}


__device__ __host__ inline float
cnorm(const cuFloatComplex &a)
{
	return sqrt(a.x * a.x + a.y * a.y);
}


template <typename T>
__device__ __host__ int
sgn(const T val)
{
	    return (val > T(0)) - (val < T(0));
}


__device__ __host__ inline cuFloatComplex
csqrt(const cuFloatComplex &a)
{
	float norm = cnorm(a);
	float frac = 1.0 / M_SQRT2;

	cuFloatComplex result;
	result.x = frac * sqrt(norm + a.x);
	result.x = frac * sqrt(norm - a.x) * sgn(a.y);
	return result;
}


__device__ __host__ inline cuFloatComplex&
csqrti(cuFloatComplex &a)
{
	float r = a.x;
	float i = a.y;
	float norm = cnorm(a);
	float frac = 1.0 / M_SQRT2;

	a.x = frac * sqrt(norm + r);
	a.y = frac * sqrt(norm - r) * sgn(i);
	return a;
}


inline std::ostream& operator<< (std::ostream &os, const cuFloatComplex &c)
{
	os << "(" << c.x << "," << c.y << ")";
	return os;
}






#endif /* __CMPLXUTIL_HPP__FA512FD4_209C_4443_906F_610B0D132370 */

