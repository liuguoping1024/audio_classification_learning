import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

RECORD_SECONDS = 3
SAMPLE_RATE = 16000
WAV_FILE = 'my_record.wav'

def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print(f"\n=== 录音中... 说点什么吧（{duration}秒） ===")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    print(f"录音完成，音频形状: {audio.shape}")
    return audio, sample_rate

def save_audio(audio, sample_rate, filename=WAV_FILE):
    sf.write(filename, audio, sample_rate)
    print(f"音频已保存为: {filename}")

def play_audio(filename=WAV_FILE):
    print(f"\n=== 回放录音 ===")
    audio, sr = sf.read(filename, dtype='float32')
    print(f"回放音频: {filename}, 采样率: {sr}, 形状: {audio.shape}")
    sd.play(audio, sr)
    sd.wait()
    print("回放结束")

def plot_waveform(audio, sample_rate):
    print(f"\n=== 可视化音频波形 ===")
    t = np.arange(len(audio)) / sample_rate
    plt.figure(figsize=(10, 3))
    plt.plot(t, audio)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sample_rate):
    print(f"\n=== 可视化音频频谱 ===")
    plt.figure(figsize=(10, 4))
    Pxx, freqs, bins, im = plt.specgram(audio, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio, sr = record_audio()
    save_audio(audio, sr)
    plot_waveform(audio, sr)
    plot_spectrogram(audio, sr)
    play_audio()
    print("\n=== 测试完成 ===") 

