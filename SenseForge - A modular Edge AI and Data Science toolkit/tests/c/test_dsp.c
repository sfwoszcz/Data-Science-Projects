#include "src/c/dsp.h"
#include <stdio.h>
int main(){float x[4]={1,2,3,4};float y;dsp_rms_f32(x,4,&y);printf("rms=%f\n",y);return 0;}
