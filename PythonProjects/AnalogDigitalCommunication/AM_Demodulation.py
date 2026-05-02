'''
Omkar Vinod Kunkolienker
23B-ET-041
SEM V
TE ETC
Communication Engineering Lab
Experiment 2: To simulate Synchronous (Coherent) Detector using Python.
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

'''Parameters to perform DFT on output of Synchronous Detector (Pre LPF)'''
m = z * np.sin(2 * np.pi * Fc * t)
M=np.fft.fft(m) # Compute DFT on Multiplied Signal
N=len(M) # Number of DFT Points
M_S = 2 * np.abs(M[:N//2])/N # DFT Normalisation & Amplitude Scaling
M_S[0]/=2 # Removing the *2 from the DC component as it doesnt need to be scaled
f = np.fft.fftfreq(N, dt) # Frequency Scale (f[n]=n/N*dt)

'''Parameters for Low Pass Filter and its output'''
H = np.zeros(N) # Creating a vector H with length N with each element being 0
BW = 300 # Bandwidth of Filter
H[np.abs(f) < BW] = 1   # Pass frequencies within 200 Hz
m_f = H * M # Applying the LPF to the output of Synchronous Detector
m_f_mag = 2*np.abs(m_f[:N//2])/N # DFT Normalisation and Amplitude Scaling
m_f_mag[0] /= 2 # Removing the *2 from the DC component as it doesnt need to be scaled
m_Demod = np.real(np.fft.ifft(m_f))  # Take real part of IFFT result
m_Demod[:] -= Vc/2 # Removing DC offset from output

'''Plotting Modulating Signal v/s Time'''
plt.figure(figsize=(10, 6))
plt.suptitle("Two-tone Modulating Signal v/s Time", fontsize=15, fontweight='bold')
plt.subplot(2, 2, 1)
plt.plot(t, x1) # Plotting monotone 1
plt.title('Monotone 1 v/s Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.1)

plt.subplot(2, 2, 2)
plt.plot(t, x2) # Plotting monotone 2
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

'''Plotting DFT of output of Synchronous Detector (Pre LPF) '''
plt.figure(figsize=(9, 3))
plt.suptitle("DFT of Output of Synchronous Detector (Pre LPF)", fontsize=15, fontweight='bold')
plt.plot(f[:N//2], M_S) # Plotting Single Sided DFT of synchronous detector
plt.title('DFT of Synchronous Detector (Pre LPF)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,2000)
plt.xticks(np.arange(0, 2001, 100))

plt.savefig("PRELPF.png", dpi=300)

'''Plotting Frequency Reponse of LPF'''
plt.figure(figsize=(9, 3))
plt.suptitle("Frequency Response of Low Pass FIlter", fontsize=15, fontweight='bold')
plt.plot(f[:N//2], H[:N//2]) # Plotting Single Sided Frequency Reponse of LPF
plt.title('Frequency Reponse of Low Pass FIlter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,1000)
plt.xticks(np.arange(0, 1001, 100))

plt.savefig("FResp.png", dpi=300)

'''Plotting DFT of output of Synchronous Detector (Post LPF) '''
plt.figure(figsize=(9, 3))
plt.suptitle("DFT of Output of Synchronous Detector (Post LPF)", fontsize=15, fontweight='bold')
plt.plot(f[:N//2], m_f_mag) # Plotting Single Sided DFT of the Demodulated Signal
plt.title('DFT of Synchronous Detector (Post LPF)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,2000)
plt.xticks(np.arange(0, 2001, 100))

plt.savefig("POSTLPF.png", dpi=300)

'''Plotting the Demodulated Signal'''
plt.figure(figsize=(9, 3))
plt.suptitle("Demodulated Signal v/s Time", fontsize=15, fontweight='bold')
plt.plot(t, m_Demod, label='Demodulated Signal') # Plotting two tone Demodulated Signal
plt.plot(t, x, '--', color='black', label='Original Modulating Signal') # Plotting original two tone ModulationSignal
plt.title('Deodulated Signal v/s Time')
plt.xlabel('Time') 
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(0,0.1)

plt.savefig("DEMOD.png", dpi=300)

plt.show()