%matplotlib inline
from pathlib import Path
from pycbc.waveform import get_td_waveform
import pylab
import numpy as np
import pycbc.noise
import pycbc.psd
from pycbc.filter import matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
import matplotlib.pyplot as plt
import japanize_matplotlib
import copy
import random
import time



# The color of the noise matches a PSD which you provide
flow = 20.0
delta_f = 1.0 / 16
flen = int(2048 / delta_f) + 1 
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

# Generate 32 seconds of noise at 4096 Hz
delta_t = 1.0 / 4096


# num,path,


e = -1
f = 1

num = 1000
np.random.seed()

spin1 = (f - e) * np.random.rand(num) + e
spin2 = (f - e) * np.random.rand(num) + e


time_sta = time.time()
s=[]
snrlist=[]
for i,s1,s2 in zip(range(num+1),spin1,spin2):
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                         mass1=36,
                         mass2=29,
                         spin1z= s1,
                         spin2z= s2,
                         delta_t=1.0/4096,
                         f_lower=20,
                         distance=500)
    seff = (36*s1+29*s2)/(36+29)
    s.append(seff)
    hp.start_time = 0
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
    ts = pycbc.noise.noise_from_psd(len(hp.sample_times), delta_t, psd, seed=127)
    hp2 = hp +ts
    times,freqs,power = hp2.qtransform(0.001, logfsteps = 100,
                                             qrange=(8,8),
                                             frange=(20,1024),)

    
    hp1 = copy.deepcopy(hp)
    hp1.resize(4096*8)
    ts = pycbc.noise.noise_from_psd(len(hp1.sample_times), delta_t, psd, seed=127)
    ts.start_time=hp.start_time
    hp4 = hp1 + ts
    segment = 4
    psd = hp4.psd(segment)
    psd = interpolate(psd, hp4.delta_f)
    psd = inverse_spectrum_truncation(psd, int((segment) * hp4.sample_rate),     low_frequency_cutoff=30)


    snr = abs(matched_filter(hp1,hp4,
                     psd=psd, low_frequency_cutoff=30))

    snr = snr.cyclic_time_shift(abs(hp1.start_time))
    snr.start_time = hp1.start_time


    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    snrp = abs(snrp)
    snrlist.append(snrp)
    
    
time_end = time.time()
tim = time_end- time_sta

print(tim)



plt.xlabel('有効スピン')
plt.ylabel('信号対雑音比')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
plt.scatter(s,snrlist,s = 3)