#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}


void sqrtAvx2(int N,
    float initialGuess,
    float values[],
    float output[])
{
    static const float kThreshold = 0.00001f;

    __m256 initialGuessVec = _mm256_set1_ps(initialGuess);
    __m256 zeros = _mm256_set1_ps(0.f);
    __m256 ones = _mm256_set1_ps(1.f);
    __m256 thresholds = _mm256_set1_ps(kThreshold);
    __m256 three = _mm256_set1_ps(3.f);
    __m256 half = _mm256_set1_ps(0.5f);

    int i = 0;
    // Process 8 elements at a time
    for (; i <= N - 8; i += 8) {
        
        __m256 x = _mm256_loadu_ps(values + i);
        __m256 guessVec = initialGuessVec;
        
        __m256 errors = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guessVec, guessVec), x), ones);
        __mmask8 negativeErrorsMask = _mm256_cmp_ps_mask(errors, zeros, _CMP_LT_OQ);
        errors = _mm256_mask_sub_ps(errors, negativeErrorsMask, zeros, errors);
        
        // Iterate until all errors are below threshold
        int iterations = 0;
        while (iterations < 100) {  // Safety limit
            __mmask8 errorMask = _mm256_cmp_ps_mask(errors, thresholds, _CMP_GT_OQ);
            if (errorMask == 0) {  // All errors are below threshold
                break;
            }
            
            // Newton-Raphson: guess = (3.f * guess - x * guess^3) * 0.5f
            // Compute guess^3
            __m256 guessCubed = _mm256_mul_ps(_mm256_mul_ps(guessVec, guessVec), guessVec);
            // Compute x * guess^3
            __m256 xTimesGuessCubed = _mm256_mul_ps(x, guessCubed);
            // Compute 3 * guess
            __m256 threeTimesGuess = _mm256_mul_ps(three, guessVec);
            // Compute (3 * guess - x * guess^3) * 0.5
            guessVec = _mm256_mul_ps(_mm256_sub_ps(threeTimesGuess, xTimesGuessCubed), half);

            // Recalculate error: guess^2 * x - 1
            __m256 guessSquared = _mm256_mul_ps(guessVec, guessVec);
            errors = _mm256_sub_ps(_mm256_mul_ps(guessSquared, x), ones);
            // Take absolute value
            negativeErrorsMask = _mm256_cmp_ps_mask(errors, zeros, _CMP_LT_OQ);
            errors = _mm256_mask_sub_ps(errors, negativeErrorsMask, zeros, errors);
            
            iterations++;
        }

        // Store result: x * guess
        __m256 result = _mm256_mul_ps(x, guessVec);
        _mm256_storeu_ps(output + i, result);
    }

    // Handle remaining elements with serial code
    for (; i < N; i++) {
        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

