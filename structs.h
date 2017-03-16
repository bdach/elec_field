typedef struct bounds {
	double x_scale;
	double y_scale;
	double x_min;
	double y_min;
	unsigned width;
	unsigned height;
} bounds_t;

typedef struct point_charge {
	double x;
	double y;
	double charge;
} point_charge_t;

typedef struct intensity {
	double x;
	double y;
} intensity_t;
