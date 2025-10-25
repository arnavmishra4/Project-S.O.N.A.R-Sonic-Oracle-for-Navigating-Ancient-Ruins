import numpy as np
import soundfile as sf
from scipy.signal import resample

def midi_to_hz(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_sine(freq, duration, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def adsr_envelope(signal, sr, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    total_len = len(signal)
    env = np.ones(total_len)
    a = int(sr * attack)
    d = int(sr * decay)
    r = int(sr * release)
    s = total_len - (a + d + r)
    env[:a] = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, sustain, d)
    env[a+d:a+d+s] = sustain
    env[-r:] = np.linspace(sustain, 0, r)
    return signal * env

def normalize_audio(signal):
    return signal / (np.max(np.abs(signal)) + 1e-10)

def write_audio(filename, signal, sr=44100):
    sf.write(filename, normalize_audio(signal), sr)

def resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample audio to a target sample rate."""
    num_samples = int(len(audio_array) * target_sr / orig_sr)
    return resample(audio_array, num_samples)
