import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import resampy
import csv
import os
import soundfile as sf

AUDIO_FILE = 'my_record.wav'

# 检查音频文件是否存在
if not os.path.exists(AUDIO_FILE):
    print(f'错误：音频文件 {AUDIO_FILE} 不存在，请先录音！')
    exit(1)

print(f'=== 加载音频文件: {AUDIO_FILE} ===')
audio, sample_rate = sf.read(AUDIO_FILE, dtype='float32')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # 转为单声道
print(f'音频采样率: {sample_rate}, 形状: {audio.shape}')

print("=== 加载YAMNet模型 ===")
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet模型加载成功!")

print("=== 加载类别名称 ===")
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = {}
with open(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names[int(row['index'])] = row['display_name']
print(f"加载了 {len(class_names)} 个音频类别")

# 重采样到16kHz
if sample_rate != 16000:
    print(f'重采样: {sample_rate} -> 16000 Hz')
    audio = resampy.resample(audio, sample_rate, 16000)
    sample_rate = 16000

print("=== YAMNet分类中 ===")
scores, embeddings, spectrogram = model(audio)
scores = scores.numpy()
mean_scores = np.mean(scores, axis=0)
top_indices = np.argsort(mean_scores)[-5:][::-1]
print("\n=== Top5类别 ===")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. {class_names[idx]}: {mean_scores[idx]:.4f}")

print("\n=== 测试完成 ===")
    
