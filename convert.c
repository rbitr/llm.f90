#include <fp16.h>

uint16_t float_to_half(float x) {
	return fp16_ieee_from_fp32_value(x);
}

float half_to_float(uint16_t h) {
	return fp16_ieee_to_fp32_value(h);
}
