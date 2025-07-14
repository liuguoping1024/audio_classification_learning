import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import tensorflow as tf
import resampy
import csv
import time

def record_audio(duration=2, sample_rate=16000):
    print(f"\n=== 录音中... 说点什么吧（{duration}秒） ===")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    print(f"录音完成，音频形状: {audio.shape}")
    return audio, sample_rate

def load_yamnet_model():
    print("=== 加载YAMNet模型 ===")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet模型加载成功!")
    return model

def load_class_names(model):
    print("=== 加载类别名称 ===")
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = {}
    with open(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names[int(row['index'])] = row['display_name']
    print(f"加载了 {len(class_names)} 个音频类别")
    return class_names

def predict_audio(model, audio, sample_rate, class_names):
    print("\n=== YAMNet分类中 ===")
    if sample_rate != 16000:
        audio = resampy.resample(audio, sample_rate, 16000)
        sample_rate = 16000
    scores, embeddings, spectrogram = model(audio)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0)
    top_indices = np.argsort(mean_scores)[-5:][::-1]
    print("\n=== 录音片段的Top5类别 ===")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {class_names[idx]}: {mean_scores[idx]:.4f}")

if __name__ == "__main__":
    try:
        import sounddevice
    except ImportError:
        print("请先安装 sounddevice: pip install sounddevice")
        exit(1)
    model = load_yamnet_model()
    class_names = load_class_names(model)
    audio, sr = record_audio(duration=2, sample_rate=16000)
    predict_audio(model, audio, sr, class_names)
    print("\n=== 测试完成 ===") 

