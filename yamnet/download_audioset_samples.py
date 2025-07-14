import os
import requests

BASE_URL = "https://storage.googleapis.com/audioset/"
FILES = [
    "golden_whistle.wav",
    "miaow_16k.wav",
    "speech_whistling2.wav",
    "yamalyzer/audio/accordion.wav",
    "yamalyzer/audio/acoustic-guitar.wav",
    "yamalyzer/audio/applause.wav",
    "yamalyzer/audio/bark.wav",
    "yamalyzer/audio/chewing.wav",
    "yamalyzer/audio/chime.wav",
    "yamalyzer/audio/cough.wav",
    "yamalyzer/audio/doorbell.wav",
    "yamalyzer/audio/explosion.wav",
    "yamalyzer/audio/fireworks.wav",
    "yamalyzer/audio/frogs.wav",
    "yamalyzer/audio/gong.wav",
    "yamalyzer/audio/gunfire.wav",
    "yamalyzer/audio/harmonica.wav",
    "yamalyzer/audio/hi-hat.wav",
    "yamalyzer/audio/knocking.wav",
    "yamalyzer/audio/marimba.wav",
    "yamalyzer/audio/meow.wav",
    "yamalyzer/audio/motorcycle.wav",
    "yamalyzer/audio/piano.wav",
    "yamalyzer/audio/rooster.wav",
    "yamalyzer/audio/sawtooth-wave.wav",
    "yamalyzer/audio/sine-wave.wav",
    "yamalyzer/audio/siren.wav",
    "yamalyzer/audio/speech.wav",
    "yamalyzer/audio/square-wave.wav",
    "yamalyzer/audio/tabla.wav",
    "yamalyzer/audio/telephone.wav",
    "yamalyzer/audio/thunder.wav",
    "yamalyzer/audio/triangle-wave.wav",
    "yamalyzer/audio/trumpet.wav",
    "yamalyzer/audio/typing.wav",
    "yamalyzer/audio/wind.wav",
    "yamalyzer/audio/zipper.wav",
]

SAVE_DIR = "audioset_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_file(rel_path):
    url = BASE_URL + rel_path
    local_path = os.path.join(SAVE_DIR, os.path.basename(rel_path))
    if os.path.exists(local_path):
        print(f"已存在: {local_path}")
        return
    print(f"下载: {url}")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"保存到: {local_path}")
    except Exception as e:
        print(f"下载失败: {url}，原因: {e}")

if __name__ == "__main__":
    for rel_path in FILES:
        download_file(rel_path)
    print("\n全部下载完成！") 

