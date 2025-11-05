#include "dsp.h"
#include <math.h>
void dsp_rms_f32(const float* x,int n,float* y){float s=0;for(int i=0;i<n;i++)s+=x[i]*x[i];*y=sqrtf(s/n);} 
