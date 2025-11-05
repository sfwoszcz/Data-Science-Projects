import numpy as np

def window_1d(x,size,stride):
    n=len(x)
    if size<=0 or stride<=0 or size>n: raise ValueError('Bad size/stride')
    num=1+(n-size)//stride
    out=np.empty((num,size),dtype=x.dtype)
    for i in range(num):
        s=i*stride
        out[i]=x[s:s+size]
    return out

def rms(x,axis=-1):
    return np.sqrt(np.mean(np.square(x),axis=axis))

def mean_std(x,axis=-1):
    return np.mean(x,axis=axis), np.std(x,axis=axis)

def zcr(x,axis=-1):
    sig=np.sign(x); crossings=np.sum(np.abs(np.diff(sig,axis=axis))>0,axis=axis)
    return crossings/x.shape[axis]
