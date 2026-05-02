'''
Omkar Vinod Kunkolienker
23B-ET-041
SEM V
TE ETC
Batch B
Communication Engineering Lab
Experiment 10: To simulate Binary Phase Shift Keying (PSK) using Python.
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

'''Carrier Signal Parameters'''
fc = 400                                  # Carrier Frequency (Hz)
carrier = np.cos(2 * np.pi * fc * t)      # Carrier Signal

'''Conversion of Bitstream into Bit Signal'''
N = len(t)                                # Total Number of Time Samples
bit_signal = np.zeros(N)                  # Initialize Empty Signal Array
for i, bit in enumerate(bit_stream_bi):   # Expanding Each Bit Over Its Bit Duration
    bit_signal[int(i * bit_duration * Fs) : int((i + 1) * bit_duration * Fs)] = bit

'''BPSK Signal Generation'''
bpsk_signal = bit_signal * carrier        # Binary Phase Shift Keyed Signal

'''FFT Parameters'''
bpsk_fft = np.fft.fft(bpsk_signal)        # FFT of BPSK Signal
bpsk_fft = np.fft.fftshift(np.abs(bpsk_fft) / N)  # Normalized and Centered Spectrum
f = np.fft.fftshift(np.fft.fftfreq(N, dt))        # Frequency Axis

'''Plotting Message (Bit) Signal, Carrier, and BPSK Signal v/s Time'''
plt.figure(figsize=(12, 8))
plt.suptitle("Modulating signal, carrier signal and BPSK signal v/s Time", fontsize=15, fontweight='bold')

'''Plotting Message (Bit) Signal v/s Time'''
plt.subplot(3, 1, 1)
plt.plot(t, bit_signal)
plt.title("Modulating Signal (Bipolar Bit Stream)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)

'''Plotting Carrier Signal v/s Time'''
plt.subplot(3, 1, 2)
plt.plot(t, carrier)
plt.title(f"Carrier Signal (fc = {fc} Hz)")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)

'''Plotting BPSK Signal v/s Time'''
plt.subplot(3, 1, 3)
plt.plot(t, bpsk_signal)
plt.title("BPSK Signal")
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.xlim(0,2*bit_duration)
plt.grid(True)
plt.tight_layout()
plt.savefig('BPSK_TimeDomain.png', dpi=300)

'''Plotting Frequency Spectrum of BPSK Signal'''
plt.figure(figsize=(12, 5))
plt.suptitle("Spectrum of the BPSK signal v/s Frequency", fontsize=15, fontweight='bold')
plt.plot(f, bpsk_fft)
plt.title("BPSK Spectrum Showing Sinc Envelope")
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.xticks(np.arange(-5 * fc, 5 * fc + 1, 100))
plt.xlim(-3 * fc, 3 * fc)
plt.grid(True)

'''Marking First Nulls for Nominal Bandwidth'''
nom_bw = 2 * bit_rate                    # Nominal Bandwidth = 2 × Rb
plt.axvline(fc - bit_rate, color='green', linestyle='--', label='First Null (Left)')
plt.axvline(fc + bit_rate, color='red', linestyle='--', label='First Null (Right)')

plt.annotate(str(int(fc-bit_rate)) + ' Hz', xy=(fc-bit_rate, 0.001), xytext=(100, 0.005),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
plt.annotate(str(int(fc+bit_rate)) + ' Hz', xy=(fc+bit_rate, 0.001), xytext=(600, 0.005),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.text(-1100, 0.005, 'Nominal Bandwidth = '+ str(nom_bw) + ' Hz', fontsize=12, color='black')
plt.legend()
plt.tight_layout()
plt.savefig('BPSK_Spectrum.png', dpi=300)

plt.show()
