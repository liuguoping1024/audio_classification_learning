import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow_hub as hub
import resampy
import csv
import os
import sys
import time

# 1. 选择音频文件
if len(sys.argv) > 1:
    wav_path = sys.argv[1]
    print(f'使用命令行参数指定的音频文件: {wav_path}')
else:
    wav_path = input('请输入要分析的wav文件路径: ').strip()
if not os.path.exists(wav_path):
    print(f'错误：文件 {wav_path} 不存在！')
    exit(1)

# 2. 读取音频
print(f'加载音频: {wav_path}')
audio, sample_rate = sf.read(wav_path, dtype='float32')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
print(f'采样率: {sample_rate}, 形状: {audio.shape}')

# # 3. 可视化：彩色波形和频谱
# fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# # 彩色波形
# ax = axs[0]
# t = np.arange(len(audio)) / sample_rate
# norm = mcolors.Normalize(vmin=-np.max(np.abs(audio)), vmax=np.max(np.abs(audio)))
# colors = plt.get_cmap('plasma')(norm(audio))
# ax.scatter(t, audio, c=colors, s=0.5, marker='.')
# ax.set_title('Audio Waveform (Colored by Amplitude)')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Amplitude')
# ax.grid(True, alpha=0.3)

# # 频谱
# ax2 = axs[1]
# Pxx, freqs, bins, im = ax2.specgram(audio, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
# ax2.set_title('Spectrogram')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Frequency (Hz)')
# fig.colorbar(im, ax=ax2, label='Intensity (dB)')
# plt.tight_layout()
# plt.show()

# 4. YAMNet分类
print('\n=== YAMNet分类 ===')
model = hub.load('https://tfhub.dev/google/yamnet/1')
# 获取类别文件
class_map_path = model.class_map_path().numpy().decode('utf-8') if hasattr(model, 'class_map_path') else None
class_names = {}
if class_map_path and os.path.exists(class_map_path):
    with open(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names[int(row['index'])] = row['display_name']
else:
    # 兼容TF Hub yamnet模型的标签获取方式
    class_names = list(model.class_names.numpy())
    class_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in class_names]
    class_names = {i: n for i, n in enumerate(class_names)}

if sample_rate != 16000:
    print(f'重采样: {sample_rate} -> 16000 Hz')
    audio = resampy.resample(audio, sample_rate, 16000)
    sample_rate = 16000

import tensorflow as tf
waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
start_time = time.time()
scores, embeddings, spectrogram = model(waveform)
predict_time = time.time() - start_time
scores = scores.numpy()
mean_scores = np.mean(scores, axis=0)
top_indices = np.argsort(mean_scores)[-5:][::-1]
print(f'\nYAMNet推理耗时: {predict_time:.3f} 秒')
print('\n=== Top5类别 ===')
for i, idx in enumerate(top_indices):
    print(f'  {i+1}. {class_names[idx]}: {mean_scores[idx]:.4f}')

print('\n=== 分析完成 ===') 

