#include <fp16.h>
#include <stdint.h>

uint16_t float_to_half(float x) {
	return fp16_ieee_from_fp32_value(x);
}

float half_to_float(uint16_t h) {
	return fp16_ieee_to_fp32_value(h);
}

uint8_t pack2x4(uint8_t xi0, uint8_t xi1) {

        return (xi0 & 0x0F) + ((xi1 & 0x0F) << 4); // could just assume <= 15

}

uint8_t unpack_high(uint8_t x) {

        return (x >> 4) & 0x0F;
}

uint8_t unpack_low(uint8_t x) {
        return x & 0x0F;
}


