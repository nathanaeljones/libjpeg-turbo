/*
 * jidctflt_fp.c
 * 
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1998, Thomas G. Lane.
 * Modified 2010 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2014, D. R. Commander.
 * Copyright (C) 2016, Nathanael Jones - perform reduction in linear light 
  * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains a floating-point implementation of the
 * inverse DCT (Discrete Cosine Transform).  In the IJG code, this routine
 * must also perform dequantization of the input coefficients.
 *
 * This implementation should be more accurate than either of the integer
 * IDCT implementations.  However, it may not give the same results on all
 * machines because of differences in roundoff behavior.  Speed will depend
 * on the hardware's floating point capacity.
 *
 * A 2-D IDCT can be done by 1-D IDCT on each column followed by 1-D IDCT
 * on each row (or vice versa, but it's more convenient to emit a row at
 * a time).  Direct algorithms are also available, but they are much more
 * complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on Arai, Agui, and Nakajima's algorithm for
 * scaled DCT.  Their original paper (Trans. IEICE E-71(11):1095) is in
 * Japanese, but the algorithm is described in the Pennebaker & Mitchell
 * JPEG textbook (see REFERENCES section in file README).  The following code
 * is based directly on figure 4-8 in P&M.
 * While an 8-point DCT cannot be done in less than 11 multiplies, it is
 * possible to arrange the computation so that many of the multiplies are
 * simple scalings of the final outputs.  These multiplies can then be
 * folded into the multiplications or divisions by the JPEG quantization
 * table entries.  The AA&N method leaves only 5 multiplies and 29 adds
 * to be done in the DCT itself.
 * The primary disadvantage of this method is that with a fixed-point
 * implementation, accuracy is lost due to imprecise representation of the
 * scaled quantization values.  However, that problem does not arise if
 * we use floating point arithmetic.
 */

 /*

 Baseline performance
 ndj-mbp-4:libjpeg-turbo-idct nathanael$ time build/djpeg -bmp -scale 1/8 snow.jpeg > /dev/null

real  0m0.247s
user  0m0.217s
sys 0m0.025s

Naive implementation:
ndj-mbp-4:libjpeg-turbo-idct nathanael$ time build/djpeg -bmp -scale 1/8 snow.jpeg > /dev/null

real  0m0.886s
user  0m0.852s
sys 0m0.027s

When active for channels != 2,3, using an approximate pow function

ndj-mbp-4:libjpeg-turbo-idct nathanael$ time build/djpeg -scale 1/8 snow.jpeg > /dev/null

real  0m0.309s
user  0m0.281s
sys 0m0.024s

*/


#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"               /* Private declarations for DCT subsystem */
#include "math.h"
#include "fastapprox.h"

#ifdef DCT_FLOAT_SUPPORTED


/*
 * This module is specialized to the case DCTSIZE = 8.
 */

#if DCTSIZE != 8
  Sorry, this code only copes with 8x8 DCTs. /* deliberate syntax err */
#endif


/* Dequantize a coefficient by multiplying it by the multiplier-table
 * entry; produce a float result.
 */

#define DEQUANTIZE(coef,quantval)  (((FAST_FLOAT) (coef)) * (quantval))



static inline float linear_to_srgb(float x)
{
  // Gamma correction
  // http://www.4p8.com/eric.brasseur/gamma.html#formulas
  if (x < 0.0f) return 0;
  if (x > 1.0f) return 255.0f;
  float r = 255.0f * fasterpow(x, 1.0f / 2.2f);
  //return 255.0f * (float)pow(clr, 1.0f / 2.2f);
  //printf("Linear %f to srgb %f\n", x, r);
  return r;
}

//THIS IS THE HOTSPOT - 90% of performance to be extracted is here
//https://stackoverflow.com/questions/6475373/optimizations-for-pow-with-const-non-integer-exponent
//Chebychev approximations could likely eliminate this hotspot
static inline float srgb_to_linear(float s)
{ 
  if (s > 255.0f) return 1.0f;
  if (s < 0.0f) return 0.0f;
  //return (float)pow(s / 255.0f, 2.2f);

  float r = fasterpow(s / 255.0f, 2.2f);
  //printf("Srgb %f to linear %f\n", s, r);
  return r;
}


/*
 * Perform dequantization and inverse DCT on one block of coefficients.
 */



GLOBAL(void) jpeg_idct_1_4_8_float (j_decompress_ptr cinfo, jpeg_component_info * compptr,
                 JCOEFPTR coef_block,
                 JSAMPARRAY output_buf, JDIMENSION output_col)
{
  FAST_FLOAT tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  FAST_FLOAT tmp10, tmp11, tmp12, tmp13;
  FAST_FLOAT z5, z10, z11, z12, z13;
  JCOEFPTR inptr;
  FLOAT_MULT_TYPE * quantptr;
  FAST_FLOAT * wsptr;
  FAST_FLOAT * linear_light_ptr;
  JSAMPROW outptr;
  JSAMPLE *range_limit = cinfo->sample_range_limit;
  int ctr, ctr_x, linear_light_x, linear_light_y;
  FAST_FLOAT workspace[DCTSIZE2]; /* buffers data between passes */
  FAST_FLOAT linear_light[DCTSIZE2]; /* buffers data between passes */
  #define _0_125 ((FLOAT_MULT_TYPE)0.125)

  /* Pass 1: process columns from input, store into work array. */

  inptr = coef_block;
  quantptr = (FLOAT_MULT_TYPE *) compptr->dct_table;
  wsptr = workspace;
  for (ctr = DCTSIZE; ctr > 0; ctr--) {
    /* Due to quantization, we will usually find that many of the input
     * coefficients are zero, especially the AC terms.  We can exploit this
     * by short-circuiting the IDCT calculation for any column in which all
     * the AC terms are zero.  In that case each output is equal to the
     * DC coefficient (with scale factor as needed).
     * With typical images and quantization tables, half or more of the
     * column DCT calculations can be simplified this way.
     */

    if (inptr[DCTSIZE*1] == 0 && inptr[DCTSIZE*2] == 0 &&
        inptr[DCTSIZE*3] == 0 && inptr[DCTSIZE*4] == 0 &&
        inptr[DCTSIZE*5] == 0 && inptr[DCTSIZE*6] == 0 &&
        inptr[DCTSIZE*7] == 0) {
      /* AC terms all zero */
      FAST_FLOAT dcval = DEQUANTIZE(inptr[DCTSIZE*0],
                                    quantptr[DCTSIZE*0] * _0_125);

      wsptr[DCTSIZE*0] = dcval;
      wsptr[DCTSIZE*1] = dcval;
      wsptr[DCTSIZE*2] = dcval;
      wsptr[DCTSIZE*3] = dcval;
      wsptr[DCTSIZE*4] = dcval;
      wsptr[DCTSIZE*5] = dcval;
      wsptr[DCTSIZE*6] = dcval;
      wsptr[DCTSIZE*7] = dcval;

      inptr++;                  /* advance pointers to next column */
      quantptr++;
      wsptr++;
      continue;
    }

    /* Even part */

    tmp0 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0] * _0_125);
    tmp1 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2] * _0_125);
    tmp2 = DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4] * _0_125);
    tmp3 = DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6] * _0_125);

    tmp10 = tmp0 + tmp2;        /* phase 3 */
    tmp11 = tmp0 - tmp2;

    tmp13 = tmp1 + tmp3;        /* phases 5-3 */
    tmp12 = (tmp1 - tmp3) * ((FAST_FLOAT) 1.414213562) - tmp13; /* 2*c4 */

    tmp0 = tmp10 + tmp13;       /* phase 2 */
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;

    /* Odd part */

    tmp4 = DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1] * _0_125);
    tmp5 = DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3] * _0_125);
    tmp6 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5] * _0_125);
    tmp7 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7] * _0_125);

    z13 = tmp6 + tmp5;          /* phase 6 */
    z10 = tmp6 - tmp5;
    z11 = tmp4 + tmp7;
    z12 = tmp4 - tmp7;

    tmp7 = z11 + z13;           /* phase 5 */
    tmp11 = (z11 - z13) * ((FAST_FLOAT) 1.414213562); /* 2*c4 */

    z5 = (z10 + z12) * ((FAST_FLOAT) 1.847759065); /* 2*c2 */
    tmp10 = z5 - z12 * ((FAST_FLOAT) 1.082392200); /* 2*(c2-c6) */
    tmp12 = z5 - z10 * ((FAST_FLOAT) 2.613125930); /* 2*(c2+c6) */

    tmp6 = tmp12 - tmp7;        /* phase 2 */
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 - tmp5;

    wsptr[DCTSIZE*0] = tmp0 + tmp7;
    wsptr[DCTSIZE*7] = tmp0 - tmp7;
    wsptr[DCTSIZE*1] = tmp1 + tmp6;
    wsptr[DCTSIZE*6] = tmp1 - tmp6;
    wsptr[DCTSIZE*2] = tmp2 + tmp5;
    wsptr[DCTSIZE*5] = tmp2 - tmp5;
    wsptr[DCTSIZE*3] = tmp3 + tmp4;
    wsptr[DCTSIZE*4] = tmp3 - tmp4;

    inptr++;                    /* advance pointers to next column */
    quantptr++;
    wsptr++;
  }

  /* Pass 2: process rows from work array, store into output array. */

  wsptr = workspace;
  linear_light_ptr = linear_light;
  for (ctr = 0; ctr < DCTSIZE; ctr++) {
    /* Rows of zeroes can be exploited in the same way as we did with columns.
     * However, the column calculation has created many nonzero AC terms, so
     * the simplification applies less often (typically 5% to 10% of the time).
     * And testing floats for zero is relatively expensive, so we don't bother.
     */

    /* Even part */

    /* Apply signed->unsigned and prepare float->int conversion */
    z5 = wsptr[0] + ((FAST_FLOAT) CENTERJSAMPLE + (FAST_FLOAT) 0.5);
    tmp10 = z5 + wsptr[4];
    tmp11 = z5 - wsptr[4];

    tmp13 = wsptr[2] + wsptr[6];
    tmp12 = (wsptr[2] - wsptr[6]) * ((FAST_FLOAT) 1.414213562) - tmp13;

    tmp0 = tmp10 + tmp13;
    tmp3 = tmp10 - tmp13;
    tmp1 = tmp11 + tmp12;
    tmp2 = tmp11 - tmp12;

    /* Odd part */

    z13 = wsptr[5] + wsptr[3];
    z10 = wsptr[5] - wsptr[3];
    z11 = wsptr[1] + wsptr[7];
    z12 = wsptr[1] - wsptr[7];

    tmp7 = z11 + z13;
    tmp11 = (z11 - z13) * ((FAST_FLOAT) 1.414213562);

    z5 = (z10 + z12) * ((FAST_FLOAT) 1.847759065); /* 2*c2 */
    tmp10 = z5 - z12 * ((FAST_FLOAT) 1.082392200); /* 2*(c2-c6) */
    tmp12 = z5 - z10 * ((FAST_FLOAT) 2.613125930); /* 2*(c2+c6) */

    tmp6 = tmp12 - tmp7;
    tmp5 = tmp11 - tmp6;
    tmp4 = tmp10 - tmp5;

    /*Convert values to linear light*/
    linear_light_ptr[0] = srgb_to_linear(tmp0 + tmp7);
    linear_light_ptr[7] = srgb_to_linear(tmp0 - tmp7);
    linear_light_ptr[1] = srgb_to_linear(tmp1 + tmp6);
    linear_light_ptr[6] = srgb_to_linear(tmp1 - tmp6);
    linear_light_ptr[2] = srgb_to_linear(tmp2 + tmp5);
    linear_light_ptr[5] = srgb_to_linear(tmp2 - tmp5);
    linear_light_ptr[3] = srgb_to_linear(tmp3 + tmp4);
    linear_light_ptr[4] = srgb_to_linear(tmp3 - tmp4);
    
    linear_light_ptr += DCTSIZE;
    wsptr += DCTSIZE;           /* advance pointer to next row */
  }
  

  //Downscale and set output values
  //Inlining and permitting those 4 loops to be unrolled
  //Didn't actually help too much, but it probably will
  //On less advanced compilers.
#define SCALE_DOWN(target_size, input_pixels_window) \
  for (ctr = 0; ctr < target_size; ctr++) { \
    outptr = output_buf[ctr] + output_col; \
    for (ctr_x = 0; ctr_x < target_size; ctr_x++){ \
      linear_light_ptr = &linear_light[ctr * input_pixels_window * DCTSIZE \
                                        + ctr_x * input_pixels_window]; \
      float sum = 0;                                                    \
      for (linear_light_y = 0; linear_light_y < input_pixels_window; linear_light_y++){ \
        for (linear_light_x = 0; linear_light_x < input_pixels_window; linear_light_x++){ \
          sum += linear_light_ptr[linear_light_x]; \
        } \
        linear_light_ptr += DCTSIZE; \
      }  \
      outptr[ctr_x] = range_limit[((int)linear_to_srgb(sum / (float)(input_pixels_window * input_pixels_window))) & RANGE_MASK]; \
    } \
  } \


  if (compptr->DCT_scaled_size == 1) {
    SCALE_DOWN(1, 8)
  } else if (compptr->DCT_scaled_size == 2) {
    SCALE_DOWN(2, 4)
  } else if (compptr->DCT_scaled_size == 4) {
    SCALE_DOWN(4, 2)
  } else {
    exit (42);
  }
}


#endif /* DCT_FLOAT_SUPPORTED */
