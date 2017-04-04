#ifndef COMPUTE_H
#define COMPUTE_H

#include "stdint.h"
#include <vector>

// floats, struct of arrays
typedef struct bounds {
	float x_scale;
	float y_scale;
	float x_min;
	float y_min;
	unsigned width;
	unsigned height;
} bounds_t;

typedef struct point_charge {
	float x;
	float y;
	float charge;
} point_charge_t;

typedef struct intensity {
	float x;
	float y;
} intensity_t;
#endif
