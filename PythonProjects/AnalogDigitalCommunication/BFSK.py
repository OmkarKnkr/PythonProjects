'''
Omkar Vinod Kunkolienker
23B-ET-041
SEM V
TE ETC
Batch B
Communication Engineering Lab
Experiment 9: To simulate Binary Frequency Shift Keying (FSK) using Python.
'''

'''Library Imports'''
import numpy as np
import matplotlib.pyplot as plt

'''Sampling, Time Axis and Message Bitstream Parameters'''
Fs = 1000000                              # Sampling Frequency (Hz)
dt = 1 / Fs                               # Sampling Interval (s)
bit_stream_uni = [0,1]                    # Bit Stream
Nb = 100                                  # Total Number of Bits
bit_rate = 100                            # Bit Rate
bit_duration = 1/bit_rate                 # Bit Duration
stop_time = Nb*bit_duration               # Stop Time
t = np.arange(0, stop_time, dt)           # Time Axis

'''Conversion of Bitstream to Bipolar Format'''
bit_stream_bi = []                        # Bipolar Bitstream Initialization
for bit in bit_stream_uni:                # 0 → -1, 1 → +1 Mapping
    bit_stream_bi.append(1 if bit == 1 else -1)

'''Conversion of Bitstream into Bit Signal'''
N=len(t)                                  # Total Number of Time Samples
bit_signal = np.zeros(N)                  # Initialize Empty Signal Array
for i, bit in enumerate(bit_stream_bi):   # Expanding Each Bit Over Its Bit Duration
    bit_signal[int(i*bit_duration*Fs): int((i+1)*bit_duration*Fs)] = bit

'''Minimum deviation = 1/(2*Tb) = 1/(2*0.01) = 25 
To maintain orthogonality deviation = n/(2*Tb) n=1,2,3,...
In this case n=12 ie deviation = 12*25 = 300'''

'''Carrier Signal Parameters'''
omega = 300                               # Frequency Deviation (Hz)
fc=600                                    # Carrier Frequency (Hz)
fh=fc+omega                               # High Frequency for Bit '1'
fl=fc-omega                               # Low Frequency for Bit '0'
carrier=np.cos(2*np.pi*fc*t)              # Original Frequency Carrier
h_carrier=np.cos(2*np.pi*fh*t)            # High Frequency Carrier
l_carrier=np.cos(2*np.pi*fl*t)            # Low Frequency Carrier

'''Splitting Bit Signal into High bits and Low bits'''
ph = (bit_signal == 1).astype(float)      # High Bit Mask (1 for '1', 0 otherwise)
pl = (bit_signal == -1).astype(float)     # Low Bit Mask (1 for '0', 0 otherwise)

h_bfsk = ph * h_carrier                   # High Frequency Component
l_bfsk = pl * l_carrier                   # Low Frequency Component
v_bfsk = h_bfsk + l_bfsk                  # Combined BFSK Signal

'''FFT Parameters'''
v_bfsk_f = np.fft.fft(v_bfsk)             # FFT of BFSK Signal
v_bfsk_f = np.fft.fftshift(np.abs(v_bfsk_f))/N   # Normalized and Centered Spectrum
f = np.fft.fftshift(np.fft.fftfreq(N,dt))        # Frequency Axis

'''Bitstream Plots'''
plt.figure(figsize=(12, 5))
plt.suptitle("Modulating bitstream, high bits and low bits in binary", fontsize=15, fontweight='bold')

plt.subplot(3,1,1)
plt.title("Modulating Signal in binary (Bipolar Bit Stream)")
plt.plot(t,bit_signal)
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,2)
plt.title("High bits in binary")
plt.plot(t,ph)
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,3)
plt.title("Low bits in binary")
plt.plot(t,pl)
plt.xlim(0,2*bit_duration)
plt.grid(True)
plt.tight_layout()
plt.savefig('BFSK_Bits_TimeDomain.png', dpi=300)

'''Carrier Plots'''
plt.figure(figsize=(12, 5))
plt.suptitle("Original, High and Low frequency carriers", fontsize=15, fontweight='bold')

plt.subplot(3,1,1)
plt.title("Original Carrier v/s time at 600 Hz")
plt.plot(t,carrier)
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,2)
plt.title("High frequency carrier v/s time at 900 Hz")
plt.plot(t,h_carrier)
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,3)
plt.title("Low frequency carrier v/s time at 300 Hz")
plt.plot(t,l_carrier)
plt.xlim(0,2*bit_duration)
plt.grid(True)
plt.tight_layout()
plt.savefig('BFSK_Carrier_TimeDomain.png', dpi=300)

'''Carrier * Bitstream plots'''
plt.figure(figsize=(12, 5))
plt.suptitle("Modulating signal, carrier signal and BPSK signal v/s Time", fontsize=15, fontweight='bold')

plt.subplot(3,1,1)
plt.plot(t,h_bfsk)
plt.title("High bit signal (Bipolar Bit Stream)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,l_bfsk)
plt.title("Low bit Signal (Bipolar Bit Stream)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t,v_bfsk)
plt.title("BFSK signal v/s time (Bipolar Bit Stream)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)
plt.tight_layout()
plt.savefig('BFSK_Signal_TimeDomain.png', dpi=300)

'''BFSK Spectrum'''
plt.figure(figsize=(12, 5))
plt.suptitle("Spectrum of the BFSK signal v/s Frequency", fontsize=15, fontweight='bold')
plt.plot(f,v_bfsk_f)
plt.xlim(-3*fc,3*fc)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.xticks(np.arange(-3 * fc, 3 * fc + 1, 200))

'''Marking First Nulls for Nominal Bandwidth'''
nom_bw = fh-fl+2*bit_rate
plt.axvline(fl - bit_rate, color='green', linestyle='--', label='First Null (Left)')
plt.axvline(fc, color='blue', linestyle='--', label='Carrier')
plt.axvline(fh + bit_rate, color='red', linestyle='--', label='First Null (Right)')
plt.annotate(str(fl-bit_rate) + ' Hz', xy=(fl-bit_rate, 0.001), xytext=(-100, 0.003),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
plt.annotate(str(fh+bit_rate) + ' Hz', xy=(fh+bit_rate, 0.001), xytext=(1200, 0.003),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
plt.annotate(str(fc) + ' Hz', xy=(fc, 0.001), xytext=(650, 0.003),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
plt.text(-1750, 0.003, 'Nominal Bandwidth =' + str(nom_bw) + ' Hz', fontsize=10, color='black')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.savefig('BFSK_Spectrum.png', dpi=300)

plt.show()
