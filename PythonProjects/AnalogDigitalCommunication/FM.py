'''
Omkar Vinod Kunkolienker
23B-ET-041
SEM V
TE ETC
Communication Engineering Lab
Experiment 4: To simulate Frequency Modulation (FM) using Python.
'''

'''Library Imports'''
import matplotlib.pyplot as plt
import numpy as np

'''Sampling and Time Axis Parametera'''
fs = 100000 # Sampling Frequency
dt = 1/fs # Sampling Interval
t = np.arange(0, 1, dt) # Time Axis 

'''Carrier Signal Parameters'''
fc = 10000 # Carrier Frequency
Vc = 10 # Carrier Amplitude
vc = Vc * np.cos(2*np.pi*fc*t) # Carrier Signal

'''Single tone message signal parameters'''
fm = 1000 # Message signal frequency
Vm = 5 # Message signal amplitude
mt = Vm * np.cos(2*np.pi*fm*t) # Message Signal

'''FM Modulation Indices to be tested'''
beta_1 = 2.4 # Modulation Index 1
beta_2 = 5.5 # Modulation Index 2
beta_3 = 6 # Modulation Index 3

'''Frequency Modulated Signald for different beta's'''
vfm_1 = Vc * np.cos(2*np.pi*fc*t + beta_1*np.sin(2*np.pi*fm*t)) # FM signal for beta = 0.5
vfm_2 = Vc * np.cos(2*np.pi*fc*t + beta_2*np.sin(2*np.pi*fm*t)) # FM Signal for beta = 3
vfm_3 = Vc * np.cos(2*np.pi*fc*t + beta_3*np.sin(2*np.pi*fm*t)) # FM Signal for beta = 6
 
'''FFT Parameters'''
N = len(t)
f = np.fft.fftfreq(N,dt) # Frequency Axis

'''FFT Parameters for FM signals'''
VFM_1 = np.fft.fft(vfm_1) # FFT of 1st FM signal
VFM_1_S = 2*np.abs(VFM_1[:N//2])/N # Normalising and scaling for single sided FFT

VFM_2 = np.fft.fft(vfm_2) # FFT of 2nd FM signal
VFM_2_S = 2*np.abs(VFM_2[:N//2])/N # Normalising and scaling for single sided FFT

VFM_3 = np.fft.fft(vfm_3) # FFT of 3rd FM Signal
VFM_3_S = 2*np.abs(VFM_3[:N//2])/N # Normalising and Scaling for single sided FFT

'''Plotting Modulating Signal v/s Time'''
plt.figure(figsize=(9,4))
plt.suptitle("Message Signal v/s Time", fontsize=15, fontweight='bold')
plt.plot(t, mt)
plt.title("Message Signal m(t)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim(0,0.01)
plt.grid(True)
plt.tight_layout()
plt.savefig('FM_Message.png', dpi = 300)

'''Plotting Carrier Signal v/s Time'''
plt.figure(figsize=(9,4))
plt.suptitle("Carrier Signal v/s Time", fontsize=15, fontweight='bold')
plt.plot(t, vc)
plt.title("Carrier Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim(0,0.01) 
plt.grid(True)
plt.tight_layout()
plt.savefig('FM_Carrier.png', dpi = 300)

'''Plotting FM Signals v/s Time'''
plt.figure(figsize=(12,7))
plt.suptitle("Tone Modulated FM Signal v/s Time", fontsize=15, fontweight='bold')
plt.subplot(3,1,1)
plt.plot(t, vfm_1) # Plotting the 1st FM Signal
plt.title("FM Signal for beta = 0.5")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.xlim(0,0.01)
plt.grid(True)
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t, vfm_2) # Plotting the 2nd FM Signal
plt.title("FM Signal for beta = 3")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.xlim(0,0.01)
plt.grid(True)
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(t, vfm_3) # Plotting the 3rd FM Signal
plt.title("FM Signal for beta = 6")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.xlim(0,0.01)
plt.grid(True)
plt.tight_layout()
plt.savefig('FM.png', dpi = 300)

'''Plotting Spectrums of FM Signals v/s Frequency'''
plt.figure(figsize=(12,7))
plt.suptitle("FM Spectrum v/s Frequency", fontsize=15, fontweight='bold')
plt.subplot(3,1,1)
plt.plot(f[:N//2], VFM_1_S) # Spectrum of 1st FM Signal
plt.title("FM Spectrum for beta = 0.5")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0,20000)
plt.xticks(np.arange(0, 20001, 1000))
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(f[:N//2], VFM_2_S) # Spectrum of 2nd FM Signal
plt.title("FM Spectrum for beta = 3")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0,20000)
plt.xticks(np.arange(0, 20001, 1000))
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(f[:N//2], VFM_3_S) # Spectrum of 3rd FM Signal
plt.title("FM Spectrum for beta = 6")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0,20000)
plt.xticks(np.arange(0, 20001, 1000))
plt.tight_layout()
plt.savefig('SFM.png', dpi = 300)

plt.show()
