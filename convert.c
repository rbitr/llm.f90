#include <stdint.h>
#include <immintrin.h>

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

float dot_product_fp16_fp32(const float* input_fp32, const uint16_t* input_fp16, int size) {
    __m256 sum = _mm256_setzero_ps(); // Initialize the sum to zero

    for (size_t i = 0; i < size; i += 8) {
        // Load 8 half-precision floats into __m128i and convert them to single-precision
        __m128i half_data = _mm_loadu_si128((__m128i const*)(input_fp16 + i));
        __m256 single_data = _mm256_cvtph_ps(half_data);

        // Load 8 single-precision float elements
        __m256 single_data2 = _mm256_loadu_ps(input_fp32 + i);

        // Multiply and accumulate
        sum = _mm256_add_ps(sum, _mm256_mul_ps(single_data, single_data2));
    }

    // Horizontal sum of the resulting vector
    __m256 hsum = _mm256_hadd_ps(sum, sum);
    __m128 hsum_low = _mm256_extractf128_ps(hsum, 0);
    __m128 hsum_high = _mm256_extractf128_ps(hsum, 1);
    hsum_low = _mm_add_ps(hsum_low, hsum_high);
    hsum_low = _mm_hadd_ps(hsum_low, hsum_low);
    hsum_low = _mm_hadd_ps(hsum_low, hsum_low);

    return _mm_cvtss_f32(hsum_low); // Extract the lower element as the final dot product result
}

float dot_product_fp16_fp32_v2(const float* input_fp32, const uint16_t* input_fp16, int size) {
    __m256 sum = _mm256_setzero_ps(); // Initialize the sum to zero

    for (size_t i = 0; i < size; i += 8) {
        // Load 8 half-precision floats into __m128i and convert them to single-precision
        __m128i half_data = _mm_loadu_si128((__m128i const*)(input_fp16 + i));
        __m256 single_data = _mm256_cvtph_ps(half_data);

        // Load 8 single-precision float elements
        __m256 single_data2 = _mm256_loadu_ps(input_fp32 + i);

        // Multiply and accumulate
        //sum = _mm256_add_ps(sum, _mm256_mul_ps(single_data, single_data2));
	sum = _mm256_fmadd_ps(single_data, single_data2, sum);
    }

    // Horizontal sum of the resulting vector
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);

    return _mm_cvtss_f32(sum_low); // Extract the lower element as the final dot product result
}
