//#include <fp16.h>
#include <stdint.h>
#include <immintrin.h>

uint16_t float_to_half(float x) {
	//return fp16_ieee_from_fp32_value(x);
	return 0;
}


static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

float half_to_float(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);	
	//return fp16_ieee_to_fp32_value(h);
}

void half_to_float_array_simd(const uint16_t* input, float* output, int size) {
    for (size_t i = 0; i < size; i += 8) {
        // Load 8 half-precision floats into __m128i
        __m128i half_data = _mm_loadu_si128((__m128i const*)(input + i));

        // Convert lower 4 half-precision floats to single-precision
        __m128 single_lo = _mm_cvtph_ps(half_data);

        // Convert upper 4 half-precision floats to single-precision
        __m128 single_hi = _mm_cvtph_ps(_mm_unpackhi_epi64(half_data, half_data));

        // Store the results in the output array
        _mm_storeu_ps(output + i, single_lo);
        _mm_storeu_ps(output + i + 4, single_hi);
    }
}

