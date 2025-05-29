# YouTube视频下载与音频处理工具

这个项目提供了一套工具，用于从YouTube下载视频、提取音频、智能分割长音频、自动转录文本，并可选择使用OpenAI进行内容综述。

## 功能特点

- **视频下载**：使用yt-dlp从YouTube下载视频并提取音频
- **智能分割**：基于静音检测自动将长音频分割成小片段
- **自动转录**：使用Whisper/MLX-Whisper模型将音频转录为文本
- **语言检测**：自动检测音频语言并适配转录
- **内容综述**：可选使用OpenAI API对转录内容进行智能总结
- **断点续传**：智能检测已下载文件和已生成文本，避免重复工作
- **跨平台支持**：Windows/Linux使用Whisper，macOS使用MLX-Whisper

## 安装说明

### 1. 安装依赖库

#### 使用uv管理本项目
首先确保本路径下venv目录是干净的
```
rm -rf venv .venv
```
然后指定版本安装venv环境
```
uv venv --python python3.12
```
然后激活环境
```
source .venv/bin/activate
```
然后安装依赖库
```
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```
需要补充下载yt—dlp
```
uv pip install yt-dlp==2025.5.28.232948.dev0
```



#### 使用pip安装

```bash
pip install -r requirements.txt
```

使用extract_cookies，这时候会弹出一个浏览器窗口，在这个窗口中登录你的账号，登录成功后，关闭浏览器窗口，回到命令行窗口，你会发现命令行窗口已经输出了cookies信息。
浏览器状态存储在user_data里面，cookies信息存储在cookies.txt里面。

这个时候可以使用：
```
yt-dlp --cookies cookies.txt --audio-format mp3 -x -o "input.mp3"  https://www.youtube.com/watch?v=n4SStAx1D3M 
```
来下载视频字幕了。
使用cosyvoice
```
docker run -it -v $PWD/cosyvoice_models/:/app/models/ --runtime=nvidia harryliu888/cosyvoice python webui.py
docker run -it -v $PWD/cosyvoice_models/:/app/models/ --runtime=nvidia harryliu888/cosyvoice /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
```
使用extract_and_download可以一次性下载视频