# Audio 分类学习项目

这是一个用于研究和实践**音频分类（audio classification）**的学习型项目，基于TensorFlow、YAMNet等主流AI工具，支持音频采集、可视化、推理和批量分析。

## 项目亮点

- **音频采集与保存**：支持麦克风录音、保存为WAV文件
- **音频可视化**：支持波形图、彩色波形、频谱图等多种可视化方式
- **YAMNet分类**：集成Google YAMNet模型，支持对任意音频文件进行AI分类，输出Top5类别
- **批量音频下载与分析**：可自动下载公开音频样本，批量推理
- **命令行与交互式体验**：支持命令行参数和交互式输入

## 主要依赖

- tensorflow >= 2.16.0
- tensorflow-hub >= 0.16.1
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- soundfile >= 0.13.0
- sounddevice >= 0.5.2
- resampy >= 0.4.3
- requests >= 2.32.0

安装依赖：
```bash
pip install -r requirements.txt
```

## 主要脚本说明

- `yamnet_mic_test.py`：麦克风录音→YAMNet分类
- `yamnet_record_visualize.py`：录音、保存、回放、波形和频谱可视化
- `yamnet_test.py`：对指定音频文件（如my_record.wav）进行YAMNet分类
- `yamnet_wav_visual_predict.py`：任意wav文件的彩色波形+频谱可视化+YAMNet分类
- `download_audioset_samples.py`：批量下载AudioSet公开测试音频

## 适用人群

- 音频AI/深度学习初学者
- 需要快速搭建音频分类实验环境的开发者
- 想要体验和分析YAMNet等主流音频模型的研究者

## 运行示例

```bash
# 录音并可视化
python yamnet_record_visualize.py

# 用YAMNet对已有音频文件分类
python yamnet_wav_visual_predict.py my_record.wav

# 批量下载公开音频样本
python download_audioset_samples.py
```

---

好好学习，天天向上！ 

