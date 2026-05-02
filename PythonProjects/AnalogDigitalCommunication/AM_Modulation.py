'''
Omkar Vinod Kunkolienker
23B-ET-041
SEM V
TE ETC
Communication Engineering Lab
Experiment 1: To simulate Amplitude Modulation using Python.
'''

'''Library Imports'''
import numpy as np
import matplotlib.pyplot as plt

'''Sampling & Time axis Parameters'''
Fs = 100000  # Sampling Frequency
dt=1/Fs # Sampling Interval
stoptime = 1  
t = np.arange(0, stoptime, dt)

'''First Monotone'''
Fm1 = 100 # Frequency
Vm1 = 2 # Amplitude
x1 = Vm1 * np.sin(2 * np.pi * Fm1 * t) # First Monotone

'''Second Monotone'''
Fm2 = 200 # Frequency
Vm2 = 4 # Amplitude
x2 = Vm2 * np.sin(2 * np.pi * Fm2 * t) # Second Monotone

'''Two Tone Modulating Signal'''
x = x1 + x2 # Modulating Signal

'''Carrier Signal'''
Fc = 500 # Carrier Frequency
Vc = 10 # Carrier Signal Amplitude
y = Vc * np.sin(2 * np.pi * Fc * t)  # Carrier Signal 

'''Amplitude Modulated Signal (AM-FC)'''
z = (Vc + x) * np.sin(2 * np.pi * Fc * t)  # AM Modulated Signal

'''Positive and Negative Envelope Parameters'''
penvelope = 10 + x # Positive Envelope
nenvelope = - (10 + x) # Negative Envelope

'''Parameters to perform FFT on AM signal'''
Z = np.fft.fft(z) # DFT on AM Signal
N = len(Z) # Number of DFT Points 
f = np.fft.fftfreq(N, dt) # Frequency Scale (f[k]=k/N*dt)

'''Plotting Modulating Signal v/s Time'''
plt.figure(figsize=(10, 6))
plt.suptitle("Two-tone Modulating Signal v/s Time", fontsize=15, fontweight='bold')
plt.subplot(2, 2, 1)
plt.plot(t, x1) # Plotting tone 1
plt.title('Monotone 1 v/s Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.subplot(2, 2, 2)
plt.plot(t, x2) # Plotting tone 2
plt.title('Monotone 2 v/s Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.subplot(2, 1, 2)
plt.plot(t, x) # Plotting two tone modulating signal
plt.title('Two Tone Modulating Signal v/s Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.savefig("Modulating.png", dpi=300)

'''Plotting Carrier Signal v/s Time'''
plt.figure(figsize=(9, 3))
plt.suptitle("Carrier Signal v/s Time", fontsize=15, fontweight='bold')
plt.plot(t, y) # Plotting Carrier Signal
plt.title('Carrier Signal y(t)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.savefig("Carrier.png", dpi=300)

'''Plotting Amplitude Modulated Signal v/s Time'''
plt.figure(figsize=(9, 3))
plt.suptitle("AM Signal (AM-FC) v/s Time", fontsize=15, fontweight='bold')
plt.plot(t, z, label='AM Signal') # Plotting AM Signal
plt.plot(t, penvelope, 'r--', label='Positive Envelope') 
plt.plot(t, nenvelope, 'g--', label='Negative Envelope')
plt.title('AM Modulated Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.savefig("AM.png", dpi=300)

'''Plotting magnitude spectrum of AM Signal'''
plt.figure(figsize=(9, 3))
plt.suptitle("DFT (magnitude) of AM Signal v/s Frequency", fontsize=15, fontweight='bold')
plt.plot(f[:N//2], 2 * np.abs(Z[:N//2]) / N) # Plotting Single Sided DFT of AM signal
plt.title('Magnitude Spectrum of AM Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,1000)
plt.xticks(np.arange(0, 1001, 100))

plt.savefig("AM-FFT.png", dpi=300)

plt.show()