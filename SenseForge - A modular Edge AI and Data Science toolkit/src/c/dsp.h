#ifndef DSP_H
#define DSP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void dsp_moving_avg_q15(const int16_t* x,int n,int w,int16_t* y);
void dsp_rms_q15(const int16_t* x,int n,int w,int16_t* y);
void dsp_moving_avg_f32(const float* x,int n,int w,float* y);
void dsp_rms_f32(const float* x,int n,int w,float* y);
#ifdef __cplusplus
}
#endif
#endif
