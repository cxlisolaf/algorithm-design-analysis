import numpy as np
import timeit
import plotly.plotly 
import matplotlib.pyplot as plt

def DFT_slow(x):
    #Compute the discrete Fourier Transform of the 1D array x
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT_python(x):
    return np.fft.fft(x)

def FFT(x):
    #A recursive implementation of the 1D Cooley-Tukey FFT
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
 
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])
     

x = np.random.random(16)

start_time=timeit.default_timer()
DFT_slow(x)
a = timeit.default_timer()-start_time
start_time2=timeit.default_timer()
FFT(x)
b=timeit.default_timer()-start_time2

#verify the result
if np.allclose(DFT_slow(x), FFT_python(x)) == True:
    print True
else:
    print False

if np.allclose(FFT(x), FFT_python(x)) == True:
    print True
else:
    print False


n_groups = 1
y = (a)
z = (b)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.001
opacity = 0.5

rects1 = plt.bar(index, a, bar_width,
                    alpha=opacity,
                    color='b',
                    label='DFT')

rects2 = plt.bar(index + bar_width, b, bar_width,
                    alpha=opacity,
                    color='g',
                    label='recursive FFT')

plt.xlabel('time in ms')
plt.ylabel('')
plt.title('DFT vs FFT')
plt.xticks(index + bar_width, ())
plt.legend()

plt.tight_layout()
plt.show()
#time_bf = timeit.Timer(DFT_slow(x).timeit(number = 100))
 
#times_fft = timeit.Timer(FFT(x).timeit(number=1000))



