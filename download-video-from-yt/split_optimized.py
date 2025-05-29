#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube视频音频下载、智能切分、转录和综述工具

主要功能：
1. 从YouTube下载音频
2. 智能切分音频文件
3. 使用Whisper进行语音转录
4. 使用OpenAI API生成综述
5. 使用CosyVoice生成语音（可选）
6. 自动cookies刷新功能

作者: Assistant
版本: 2.0
"""

import os
import sys
import re
import json
import math
import time
import asyncio
import argparse
import platform
import tempfile
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# 第三方库导入
import torch
from pydub import AudioSegment
from pydub.silence import detect_silence
from mutagen.mp3 import MP3
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 可选依赖导入
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("警告: playwright库未安装，自动cookies重新获取功能将不可用")
    print("如需使用此功能，请运行: pip install playwright && playwright install chromium")

try:
    from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
    FISHAUDIO_AVAILABLE = True
except ImportError as e:
    FISHAUDIO_AVAILABLE = False
    print(f"警告: Fish Audio SDK未安装，--with-voice选项将不可用: {e}")
    print("如需使用此功能，请运行: pip install fish-audio-sdk")

# Fish Audio配置
FISH_AUDIO_API_KEY = os.getenv('FISH_AUDIO_API_KEY')
FISH_AUDIO_REFERENCE_ID = os.getenv('FISH_AUDIO_REFERENCE_ID', 'MODEL_ID_UPLOADED_OR_CHOSEN_FROM_PLAYGROUND')

# 配置日志系统
def setup_logging():
    """
    配置日志系统，同时输出到文件和控制台
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("running_log_for_transcribe.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('audio_processor')

# 初始化日志和环境变量
logger = setup_logging()
load_dotenv()

# ==================== 音频时长获取相关函数 ====================

def get_audio_duration_fast(file_path):
    """
    快速获取音频文件时长，优先使用mutagen库
    
    Args:
        file_path (str): 音频文件路径
    
    Returns:
        int: 音频时长（毫秒），失败返回0
    """
    try:
        audio_file = MP3(file_path)
        duration_ms = int(audio_file.info.length * 1000)
        logger.debug(f"使用mutagen获取音频时长: {duration_ms}ms")
        return duration_ms
    except Exception as e:
        logger.warning(f"使用mutagen获取时长失败: {e}，尝试使用ffprobe")
        return get_duration_with_ffprobe(file_path)

def get_duration_with_ffprobe(file_path):
    """
    使用ffprobe获取音频时长（备用方案）
    
    Args:
        file_path (str): 音频文件路径
    
    Returns:
        int: 音频时长（毫秒），失败返回估算值
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_sec = float(result.stdout.strip())
        duration_ms = int(duration_sec * 1000)
        logger.debug(f"使用ffprobe获取音频时长: {duration_ms}ms")
        return duration_ms
    except Exception as e:
        logger.warning(f"使用ffprobe获取时长失败: {e}，使用文件大小估算")
        return estimate_duration_by_filesize(file_path)

def estimate_duration_by_filesize(file_path):
    """
    根据文件大小估算音频时长（最后备用方案，精度较低）
    
    Args:
        file_path (str): 音频文件路径
    
    Returns:
        int: 估算的音频时长（毫秒）
    """
    try:
        file_size = os.path.getsize(file_path)
        # 假设平均比特率为128kbps
        estimated_duration_sec = file_size / (128 * 1024 / 8)
        duration_ms = int(estimated_duration_sec * 1000)
        logger.warning(f"使用文件大小估算音频时长: {duration_ms}ms（精度较低）")
        return duration_ms
    except Exception as e:
        logger.error(f"估算音频时长失败: {e}")
        return 0

# ==================== 音频处理相关函数 ====================

def extract_chunk_with_ffmpeg(input_path, output_path, start_ms, end_ms):
    """
    使用ffmpeg提取音频片段
    
    Args:
        input_path (str): 输入音频文件路径
        output_path (str): 输出音频文件路径
        start_ms (int): 开始时间（毫秒）
        end_ms (int): 结束时间（毫秒）
    
    Returns:
        bool: 提取成功返回True，失败返回False
    """
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ss', str(start_sec),
        '-t', str(duration_sec),
        '-c', 'copy',  # 直接复制流，速度快
        output_path, '-y'  # 覆盖输出文件
    ]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug(f"成功提取音频片段: {start_ms}ms-{end_ms}ms -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg提取失败: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr.decode()}")
        return False

# ==================== URL处理相关函数 ====================

def get_video_id_from_url(url):
    """
    从YouTube URL中提取video_id
    
    Args:
        url (str): YouTube视频URL
    
    Returns:
        str: 视频ID，提取失败返回None
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ''
        
        # 处理youtube.com格式
        if 'youtube.com' in hostname:
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get('v', [None])[0]
                if video_id:
                    logger.debug(f"从youtube.com URL提取video_id: {video_id}")
                    return video_id
        
        # 处理youtu.be格式
        elif 'youtu.be' in hostname:
            if parsed_url.path.startswith('/'):
                video_id = parsed_url.path[1:]
                if video_id:
                    logger.debug(f"从youtu.be URL提取video_id: {video_id}")
                    return video_id
    
    except Exception as e:
        logger.warning(f"解析URL提取video_id失败: {e}")
    
    # 备用方案：正则匹配
    match = re.search(r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    if match:
        video_id = match.group(1)
        logger.debug(f"使用正则表达式提取video_id: {video_id}")
        return video_id
    
    logger.error(f"无法从URL提取video_id: {url}")
    return None

# ==================== Chrome和Cookies相关函数 ====================

def get_chrome_binary_path():
    """
    获取Chrome浏览器二进制文件的路径
    优先从环境变量读取，否则使用系统默认路径
    
    Returns:
        str: Chrome浏览器可执行文件的路径
    
    Raises:
        Exception: 不支持的操作系统
    """
    # 检查环境变量
    chrome_path = os.getenv('CHROME_BINARY_PATH')
    if chrome_path and os.path.exists(chrome_path):
        logger.debug(f"使用环境变量中的Chrome路径: {chrome_path}")
        return chrome_path
    
    # 根据操作系统确定默认路径
    system = platform.system()
    default_paths = {
        'Windows': r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        'Darwin': "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        'Linux': "/usr/bin/google-chrome"
    }
    
    if system in default_paths:
        chrome_path = default_paths[system]
        logger.debug(f"使用{system}系统默认Chrome路径: {chrome_path}")
        return chrome_path
    else:
        raise Exception(f"不支持的操作系统: {system}")

def launch_chrome_with_remote_debugging(port=9222, user_data="./user_data"):
    """
    启动Chrome浏览器并开启远程调试端口
    
    Args:
        port (int): 远程调试端口号，默认为9222
        user_data (str): 用户数据目录，默认为./user_data
    
    Returns:
        subprocess.Popen: 启动的Chrome进程对象
    
    Raises:
        Exception: Chrome启动失败
    """
    try:
        chrome_path = get_chrome_binary_path()
        logger.info(f"使用Chrome路径: {chrome_path}")
        
        # 构建用户数据目录的完整路径
        user_data_path = Path(__file__).resolve().parent / user_data
        user_data_path.mkdir(parents=True, exist_ok=True)
        
        # 构建Chrome启动命令
        cmd = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_path}"
        ]
        
        # 启动Chrome进程
        logger.info(f"启动Chrome，远程调试端口: {port}")
        process = subprocess.Popen(cmd)
        
        # 等待用户登录网站
        print("请在打开的Chrome浏览器中登录YouTube网站，然后按回车键继续...")
        input()
        
        return process
        
    except Exception as e:
        logger.error(f"启动Chrome失败: {e}")
        raise

async def save_cookies_async(output_file='cookies.txt'):
    """
    异步连接到Chrome并保存cookies到文件
    
    Args:
        output_file (str): 保存cookies的文件名
    
    Returns:
        int: 成功返回0，失败返回1
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("playwright库未安装，无法自动获取cookies")
        return 1
    
    async with async_playwright() as p:
        browser = None
        try:
            logger.info("尝试连接到Chrome浏览器 (http://localhost:9222)...")
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
            logger.info("成功连接到Chrome浏览器")
            
            default_context = browser.contexts[0]
            cookies = await default_context.cookies()
            logger.info(f"获取到 {len(cookies)} 个cookies")
            
            # 转换并保存cookies
            convert_json_to_netscape_cookies(cookies, output_file)
            logger.info(f"Cookies已保存到 {output_file}")
            
            return 0
            
        except Exception as e:
            logger.error(f"保存cookies时出错: {e}")
            logger.error("请确保Chrome浏览器已通过远程调试模式启动，端口为9222")
            return 1
        finally:
            if browser:
                await browser.close()
                logger.info("已关闭与Chrome浏览器的连接")

def convert_json_to_netscape_cookies(cookies, netscape_file):
    """
    将JSON格式的cookies转换为Netscape格式
    
    Args:
        cookies (list): JSON格式的cookies列表
        netscape_file (str): 输出的Netscape格式cookies文件路径
    
    Raises:
        Exception: 转换失败
    """
    try:
        with open(netscape_file, 'w', encoding='utf-8') as f:
            # 写入Netscape cookies文件头
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# https://curl.haxx.se/docs/http-cookies.html\n")
            f.write("# This file was generated by a script\n\n")
            
            # 写入每个cookie
            for cookie in cookies:
                domain = cookie.get('domain', '')
                is_domain_wide = domain.startswith('.')
                domain_specified = 'TRUE' if is_domain_wide else 'FALSE'
                path = cookie.get('path', '/')
                secure = 'TRUE' if cookie.get('secure', False) else 'FALSE'
                
                # 处理过期时间
                expires_val = cookie.get('expires', -1)
                if not isinstance(expires_val, (int, float)) or expires_val <= 0:
                    expires = 0  # 会话cookie
                else:
                    expires = int(expires_val)
                
                name = cookie.get('name', '')
                value = cookie.get('value', '')
                
                # Netscape格式: domain\tdomain_specified\tpath\tsecure\texpiry\tname\tvalue
                f.write(f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
        
        logger.info(f"Cookies已转换为Netscape格式并保存到 {netscape_file}")
        
    except Exception as e:
        logger.error(f"转换cookies格式时出错: {e}")
        raise

def refresh_cookies_interactive(cookies_file="cookies.txt"):
    """
    交互式刷新cookies
    
    Args:
        cookies_file (str): cookies文件路径
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("playwright库未安装，无法自动刷新cookies")
        logger.info("请手动更新cookies.txt文件")
        return False
    
    chrome_process = None
    try:
        logger.info("检测到下载失败，尝试自动刷新cookies...")
        logger.info("将启动Chrome浏览器，请在浏览器中重新登录YouTube")
        
        # 启动Chrome并开启远程调试
        chrome_process = launch_chrome_with_remote_debugging()
        
        if chrome_process:
            # 运行异步函数保存cookies
            exit_code = asyncio.run(save_cookies_async(cookies_file))
            
            if exit_code == 0:
                logger.info("Cookies刷新成功！")
                return True
            else:
                logger.error("Cookies刷新失败")
                return False
        else:
            logger.error("未能启动Chrome浏览器")
            return False
    
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return False
    except Exception as e:
        logger.error(f"刷新cookies时发生错误: {e}")
        return False
    finally:
        if chrome_process:
            logger.info("正在关闭Chrome进程...")
            chrome_process.terminate()
            try:
                chrome_process.wait(timeout=5)
                logger.info("Chrome进程已关闭")
            except subprocess.TimeoutExpired:
                logger.warning("关闭Chrome进程超时，强制终止")
                chrome_process.kill()
    
    return False

# ==================== 音频下载相关函数 ====================

def download_audio(url, output_path, cookies_file="cookies.txt", auto_refresh_cookies=True):
    """
    使用yt-dlp下载音频，支持自动刷新cookies重试
    
    Args:
        url (str): 视频URL
        output_path (str): 输出路径
        cookies_file (str): cookies文件路径
        auto_refresh_cookies (bool): 是否在下载失败时自动刷新cookies
    
    Returns:
        bool: 下载成功返回True，失败返回False
    """
    def _attempt_download():
        """执行单次下载尝试"""
        logger.info(f"开始下载音频: {url} -> {output_path}")
        
        command = [
            'yt-dlp',
            '--cookies', cookies_file,
            '--audio-format', 'mp3',
            '-x',
            '-o', output_path,
            url
        ]
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"音频成功下载: {output_path}")
                return True, None, None
            else:
                return False, stdout.decode(errors='ignore'), stderr.decode(errors='ignore')
        
        except FileNotFoundError:
            logger.error("yt-dlp命令未找到，请确保已安装并配置在系统PATH中")
            return False, None, "yt-dlp command not found"
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            return False, None, str(e)
    
    # 第一次尝试下载
    success, stdout, stderr = _attempt_download()
    
    if success:
        return True
    
    # 记录第一次下载失败的错误信息
    logger.error("下载失败")
    if stdout:
        logger.error(f"yt-dlp stdout: {stdout}")
    if stderr:
        logger.error(f"yt-dlp stderr: {stderr}")
    
    # 如果启用了自动刷新cookies功能，尝试刷新cookies并重试
    if auto_refresh_cookies:
        logger.info("尝试刷新cookies后重新下载...")
        
        try:
            user_choice = input("下载失败，是否要自动刷新cookies并重试？(y/n，默认为y): ").strip().lower()
            if user_choice in ['', 'y', 'yes']:
                # 刷新cookies
                if refresh_cookies_interactive(cookies_file):
                    logger.info("Cookies刷新成功，重新尝试下载...")
                    
                    # 第二次尝试下载
                    success, stdout, stderr = _attempt_download()
                    
                    if success:
                        return True
                    else:
                        logger.error("使用新cookies下载仍然失败")
                        if stdout:
                            logger.error(f"yt-dlp stdout: {stdout}")
                        if stderr:
                            logger.error(f"yt-dlp stderr: {stderr}")
                else:
                    logger.error("Cookies刷新失败")
            else:
                logger.info("用户选择不刷新cookies")
        
        except KeyboardInterrupt:
            logger.info("用户中断操作")
        except Exception as e:
            logger.error(f"处理用户输入时发生错误: {e}")
    
    return False

# ==================== 音频切分和转录相关函数 ====================

def detect_silences_in_audio(audio_segment, min_silence_len=300, silence_thresh=-55):
    """
    检测音频中的静音段
    
    Args:
        audio_segment: pydub AudioSegment对象
        min_silence_len (int): 最小静音长度（毫秒）
        silence_thresh (int): 静音阈值（dBFS）
    
    Returns:
        list: 静音段列表，每个元素为(start_ms, end_ms)
    """
    try:
        silences = detect_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        logger.debug(f"检测到 {len(silences)} 个静音段")
        return silences
    except Exception as e:
        logger.error(f"检测静音段失败: {e}")
        return []

def find_optimal_split_point(silences, target_end_ms, seek_window_ms):
    """
    在静音段中寻找最优的切分点
    
    Args:
        silences (list): 静音段列表
        target_end_ms (int): 目标结束时间
        seek_window_ms (int): 搜索窗口大小
    
    Returns:
        int: 最优切分点时间（毫秒）
    """
    search_start = max(0, target_end_ms - seek_window_ms)
    search_end = target_end_ms + seek_window_ms
    
    best_silence_start = None
    min_diff = float('inf')
    
    for silence_start, silence_end in silences:
        if search_start <= silence_start <= search_end:
            # 优先选择目标时间之后的静音段
            if silence_start >= target_end_ms:
                diff = silence_start - target_end_ms
                if diff < min_diff:
                    min_diff = diff
                    best_silence_start = silence_start
                    break  # 找到第一个合适的就停止
            
            # 如果没有找到目标时间之后的，选择最接近的
            diff = abs(silence_start - target_end_ms)
            if diff < min_diff:
                min_diff = diff
                best_silence_start = silence_start
    
    return best_silence_start if best_silence_start is not None else target_end_ms

def load_whisper_model(system):
    """
    根据操作系统加载相应的Whisper模型
    
    Args:
        system (str): 操作系统名称
    
    Returns:
        object: Whisper模型对象，失败返回None
    """
    try:
        if system in ['Windows', 'Linux']:
            import whisper
            logger.info("加载Whisper模型...")
            model = whisper.load_model("large-v3")
            logger.info("Whisper模型加载成功")
            return model
        elif system == 'Darwin':
            import mlx_whisper
            logger.info("macOS系统，使用MLX Whisper")
            return mlx_whisper  # MLX Whisper不需要预加载模型
        else:
            logger.error(f"不支持的操作系统: {system}")
            return None
    except Exception as e:
        logger.error(f"加载Whisper模型失败: {e}")
        return None

def transcribe_audio_segment(audio_file_path, model, system, language=None):
    """
    转录单个音频片段
    
    Args:
        audio_file_path (str): 音频文件路径
        model: Whisper模型对象
        system (str): 操作系统名称
        language (str): 语言代码，None为自动检测
    
    Returns:
        str: 转录文本，失败返回空字符串
    """
    try:
        initial_prompt = "提取音频中的文字内容。" if language == "zh" else "transcribe the following audio."
        
        if system in ['Windows', 'Linux']:
            result = model.transcribe(audio_file_path, language=language, initial_prompt=initial_prompt)
        elif system == 'Darwin':
            result = model.transcribe(
                audio_file_path,
                language=language,
                initial_prompt=initial_prompt,
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
            )
        else:
            logger.error(f"不支持的操作系统: {system}")
            return ""
        
        return result.get('text', '').strip()
    
    except Exception as e:
        logger.error(f"转录音频片段失败: {e}")
        return ""

def smart_split_mp3(file_path, output_dir, target_duration_sec=60, silence_thresh=-55, 
                   min_silence_len=300, seek_window_sec=10, chunk_duration_min=20):
    """
    智能切分MP3文件并进行转录
    
    Args:
        file_path (str): 输入音频文件路径
        output_dir (str): 输出目录
        target_duration_sec (int): 目标片段时长（秒）
        silence_thresh (int): 静音阈值（dBFS）
        min_silence_len (int): 最小静音长度（毫秒）
        seek_window_sec (int): 搜索窗口大小（秒）
        chunk_duration_min (int): 主块处理时长（分钟）
    
    Returns:
        str: 转录结果文件路径，失败返回None
    """
    # 获取音频总时长
    total_length_ms = get_audio_duration_fast(file_path)
    if total_length_ms == 0:
        logger.error("无法获取音频时长，退出")
        return None
    
    logger.info(f"音频总时长: {total_length_ms/1000:.2f} 秒")
    
    # 检查操作系统并加载模型
    system = platform.system()
    model = load_whisper_model(system)
    if model is None:
        logger.error("无法加载Whisper模型")
        return None
    
    # 转换参数单位
    target_duration_ms = target_duration_sec * 1000
    seek_window_ms = seek_window_sec * 1000
    chunk_duration_ms = chunk_duration_min * 60 * 1000
    
    # 初始化变量
    output_text_file = os.path.join(output_dir, "result.txt")
    global_start_ms = 0
    part_num = 1
    detected_language = None
    
    # 创建进度条
    total_progress = tqdm(total=total_length_ms/1000, desc="总体进度", unit="秒")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            while global_start_ms < total_length_ms:
                chunk_end_ms = min(global_start_ms + chunk_duration_ms, total_length_ms)
                logger.info(f"处理主块: {global_start_ms/1000/60:.2f} - {chunk_end_ms/1000/60:.2f} 分钟")
                
                # 提取主块
                temp_main_chunk_path = os.path.join(temp_dir, f"temp_main_chunk_{part_num}.mp3")
                
                if not extract_chunk_with_ffmpeg(file_path, temp_main_chunk_path, global_start_ms, chunk_end_ms):
                    logger.warning("无法提取主块，跳过")
                    global_start_ms = chunk_end_ms
                    continue
                
                # 加载音频段
                try:
                    current_chunk_audio = AudioSegment.from_mp3(temp_main_chunk_path)
                except Exception as e:
                    logger.error(f"加载音频块失败: {e}")
                    global_start_ms = chunk_end_ms
                    continue
                
                # 检测静音段
                silences = detect_silences_in_audio(current_chunk_audio, min_silence_len, silence_thresh)
                
                # 处理当前块的分段
                chunk_internal_start_ms = 0
                while chunk_internal_start_ms < len(current_chunk_audio):
                    # 计算目标结束点
                    chunk_internal_end_candidate = min(
                        chunk_internal_start_ms + target_duration_ms,
                        len(current_chunk_audio)
                    )
                    
                    # 寻找最优切分点
                    split_point = find_optimal_split_point(
                        silences, chunk_internal_end_candidate, seek_window_ms
                    )
                    
                    # 确保切分点有效
                    if split_point <= chunk_internal_start_ms:
                        split_point = min(
                            chunk_internal_start_ms + target_duration_ms,
                            len(current_chunk_audio)
                        )
                    
                    split_point = min(split_point, len(current_chunk_audio))
                    
                    # 提取音频段
                    segment = current_chunk_audio[chunk_internal_start_ms:split_point]
                    if len(segment) == 0:
                        break
                    
                    # 导出音频段
                    base_filename = os.path.splitext(os.path.basename(file_path))[0]
                    segment_filename = os.path.join(output_dir, f"{base_filename}_part_{part_num:03d}.mp3")
                    segment.export(segment_filename, format="mp3")
                    
                    logger.debug(f"导出音频段: {segment_filename} (时长: {len(segment)/1000:.2f}秒)")
                    
                    # 转录音频段
                    transcription = transcribe_audio_segment(segment_filename, model, system, detected_language)
                    
                    if transcription:
                        # 保存转录结果
                        with open(output_text_file, 'a', encoding='utf-8') as f:
                            f.write(transcription + '\n')
                        
                        # 检测语言（仅第一次）
                        if detected_language is None and transcription:
                            # 简单的中文检测
                            if re.search(r'[\u4e00-\u9fff]', transcription):
                                detected_language = "zh"
                                logger.info("检测到中文内容")
                            else:
                                detected_language = "en"
                                logger.info("检测到英文内容")
                    
                    # 删除临时音频段文件
                    try:
                        os.remove(segment_filename)
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {e}")
                    
                    # 更新进度
                    total_progress.update(len(segment)/1000)
                    
                    # 移动到下一个段
                    chunk_internal_start_ms = split_point
                    part_num += 1
                    
                    if split_point >= len(current_chunk_audio):
                        break
                
                # 移动到下一个主块
                global_start_ms = chunk_end_ms
    
    except Exception as e:
        logger.error(f"音频切分过程中发生错误: {e}")
        return None
    finally:
        total_progress.close()
    
    logger.info(f"音频切分和转录完成，结果保存到: {output_text_file}")
    return output_text_file

# ==================== OpenAI综述相关函数 ====================

def summarize_text_with_openai(text_file_path, output_dir, model="gpt-3.5-turbo"):
    """
    使用OpenAI API对文本进行综述
    
    Args:
        text_file_path (str): 输入文本文件路径
        output_dir (str): 输出目录
        model (str): OpenAI模型名称
    
    Returns:
        str: 综述内容，失败返回None
    """
    logger.info(f"开始使用OpenAI对文本进行综述: {text_file_path}")
    
    try:
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        
        if not api_key:
            logger.error("OPENAI_API_KEY环境变量未设置")
            return None
        if not base_url:
            logger.error("BASE_URL环境变量未设置")
            return None
        
        # 优先使用环境变量中的模型
        env_model = os.getenv("MODEL_NAME")
        if env_model and env_model.strip():
            model = env_model
            logger.info(f"使用环境变量中的模型: {model}")
        else:
            logger.info(f"使用默认模型: {model}")
        
        # 读取文本内容
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            logger.warning("文本文件内容为空，无法进行综述")
            return None
        
        # 创建OpenAI客户端
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 构建提示词
        prompt_message = f'''
{content}

以上是一份优秀的口播稿件。
你是一个优秀的口播撰稿人，你写稿子的特点如下：
1、你的频道的名字是kaka乱弹。一般以"hello，欢迎来到kaka乱弹的频道"开头。
2、你面向的听众是年轻白领，你的性格是年轻、爱美、活泼、聪明。
3、你善于把握重点，擅长对复杂的问题进行形象化举例说明。
4、你的语言丰富，擅长用网络梗，不喜欢用markdown、表情包、代码块写稿
5、你的语言风格接地气，段落和内容切换自然合理，你会规避使用1、2、3...这样的罗列信息的方式来表达，而是采用自然、口语化的方式来衔接
6、你学习前面给出的内容，但会避免使用内容中具有个人风格的表达方式，避免使用内容中带有的广告、宣传推广的内容。

现在，请你学习上面的知识，按照你自己的角色风格重新写一份精彩的口播演讲稿。
'''
        
        # 调用OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_message}
            ]
        )
        
        summary = response.choices[0].message.content
        
        logger.info("OpenAI综述生成成功")
        print("\n--- OpenAI 综述结果 ---")
        print(summary)
        print("--- 综述结束 ---")
        
        # 保存综述结果
        summary_filename = f"{os.path.splitext(os.path.basename(text_file_path))[0]}_summary.txt"
        summary_file_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"综述结果已保存到: {summary_file_path}")
        return summary
    
    except Exception as e:
        logger.error(f"调用OpenAI API时发生错误: {e}")
        return None

# ==================== 语音生成相关函数 ====================

def split_text_into_sentences(text):
    """
    将文本按句子拆分，支持中英文标点
    
    Args:
        text (str): 输入文本
    
    Returns:
        list: 句子列表
    """
    # 使用正则表达式按句子分割
    sentences = re.split(r'[。！？.!?]+', text)
    # 过滤空字符串并去除首尾空格
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_voice_from_summary(summary_file_path, output_dir, 
                               reference_id=None, 
                               api_key=None):
    """
    使用Fish Audio将综述文本转换为语音
    先将文本拆分成句子，逐句生成语音，最后合并成一个完整的音频文件
    
    Args:
        summary_file_path (str): 综述文本文件路径
        output_dir (str): 输出目录
        reference_id (str): Fish Audio模型引用ID
        api_key (str): Fish Audio API密钥
    
    Returns:
        str: 生成的语音文件路径，失败返回None
    """
    if not FISHAUDIO_AVAILABLE:
        logger.error("Fish Audio SDK未安装，无法生成语音")
        return None
    
    # 使用传入的参数或环境变量
    api_key = os.getenv("FISH_AUDIO_API_KEY")
    reference_id = os.getenv("FISH_AUDIO_REFERENCE_ID") 
    
    if not api_key:
        logger.error("Fish Audio API密钥未设置，请设置FISH_AUDIO_API_KEY环境变量")
        return None
    
    if not reference_id:
        logger.error("Fish Audio模型引用ID未设置，请设置FISH_AUDIO_REFERENCE_ID环境变量")
        return None
    
    try:
        logger.info(f"开始使用Fish Audio生成语音，模型ID: {reference_id}")
        
        # 读取综述文本
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
        
        if not summary_text:
            logger.error("综述文件为空，无法生成语音")
            return None
        
        # 将文本拆分成句子
        sentences = split_text_into_sentences(summary_text)
        logger.info(f"文本已拆分为 {len(sentences)} 个句子")
        
        if not sentences:
            logger.error("文本拆分后没有有效句子")
            return None
        
        # 初始化Fish Audio会话
        logger.info("正在初始化Fish Audio会话...")
        
        # 检查是否需要使用代理
        proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        if proxy_url:
            logger.info(f"使用代理: {proxy_url}")
            # Fish Audio SDK使用base_url参数来设置代理域名
            # 如果proxy_url是完整的代理地址，我们需要将其转换为base_url格式
            if proxy_url.startswith('http'):
                # 对于HTTP代理，我们需要设置环境变量让httpx自动处理
                import httpx
                # 创建带代理的httpx客户端
                session = Session(api_key)
                # 设置代理环境变量，让底层的httpx使用代理
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
            else:
                session = Session(api_key, base_url=proxy_url)
        else:
            session = Session(api_key)
        
        # 创建临时目录存储单句语音文件
        temp_dir = os.path.join(output_dir, "temp_audio_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 逐句生成语音
        audio_segments = []
        logger.info("开始逐句生成语音...")
        i=0
        for sentence in sentences:
            if not sentence.strip():
                continue
            s_len = min(50,len(sentence))
            logger.info(f"正在生成语音: {sentence[:s_len]}...")
            
            # 生成单句语音文件路径
            segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
            i += 1
            try:
                # 使用Fish Audio生成单句语音
                with open(segment_file, "wb") as f:
                    for chunk in session.tts(TTSRequest(
                        reference_id=reference_id,
                        text=sentence 
                    )):
                        f.write(chunk)
                
                audio_segments.append(segment_file)
                logger.debug(f"第 {i} 句语音生成完成: {segment_file}")
                
            except Exception as e:
                logger.warning(f"生成第 {i} 句语音时出错: {e}，跳过该句")
                continue
        print(audio_segments)
        if not audio_segments:
            logger.error("没有成功生成任何语音片段")
            return None
        
        # 合并所有语音片段
        logger.info(f"开始合并 {len(audio_segments)} 个语音片段...")
        output_audio_file = os.path.join(output_dir, "summary_voice.mp3")
        
        try:
            # 使用pydub合并音频
            combined_audio = AudioSegment.empty()
            
            for segment_file in audio_segments:
                if os.path.exists(segment_file):
                    segment_audio = AudioSegment.from_mp3(segment_file)
                    combined_audio += segment_audio
                    # 在句子之间添加短暂停顿（500毫秒）
                    combined_audio += AudioSegment.silent(duration=500)
            
            # 导出合并后的音频
            combined_audio.export(output_audio_file, format="mp3")
            logger.info(f"语音合并完成，总时长: {len(combined_audio)/1000:.2f} 秒")
            
        except Exception as e:
            logger.error(f"合并语音文件时发生错误: {e}")
            # 如果合并失败，尝试使用ffmpeg
            logger.info("尝试使用ffmpeg合并音频文件...")
            try:
                # 创建文件列表
                file_list_path = os.path.join(temp_dir, "file_list.txt")
                with open(file_list_path, 'w', encoding='utf-8') as f:
                    for segment_file in audio_segments:
                        if os.path.exists(segment_file):
                            f.write(f"file '{segment_file}'\n")
                
                # 使用ffmpeg合并
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', file_list_path,
                    '-c', 'copy', output_audio_file, '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("使用ffmpeg成功合并音频文件")
                else:
                    logger.error(f"ffmpeg合并失败: {result.stderr}")
                    return None
                    
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg合并也失败了: {ffmpeg_error}")
                return None
        
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.debug("临时文件清理完成")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")
        
        if os.path.exists(output_audio_file):
            logger.info(f"语音文件已保存到: {output_audio_file}")
            return output_audio_file
        else:
            logger.error("最终语音文件生成失败")
            return None
    
    except Exception as e:
        logger.error(f"生成语音时发生错误: {e}")
        return None

# ==================== 主函数和命令行参数处理 ====================

def create_output_directories(video_id):
    """
    创建输出目录结构
    
    Args:
        video_id (str): 视频ID
    
    Returns:
        str: 视频专属输出目录路径
    """
    # 创建主输出目录
    main_output_dir = "output"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
        logger.debug(f"创建主输出目录: {main_output_dir}")
    
    # 创建视频专属输出目录
    video_output_dir = os.path.join(main_output_dir, video_id)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
        logger.debug(f"创建视频专属输出目录: {video_output_dir}")
    
    return video_output_dir

def check_file_exists_and_force(file_path, file_description, force_flag):
    """
    检查文件是否存在，并根据force标志决定是否跳过
    
    Args:
        file_path (str): 文件路径
        file_description (str): 文件描述
        force_flag (bool): 强制执行标志
    
    Returns:
        bool: True表示需要执行，False表示跳过
    """
    if os.path.exists(file_path) and not force_flag:
        logger.info(f"{file_description}已存在: {file_path}")
        logger.info(f"跳过{file_description}步骤。如需重新生成，请使用 --force 选项。")
        return False
    return True

def main():
    """
    主函数：处理命令行参数并执行相应的操作
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description='下载YouTube视频的音频，进行智能切分和转录，并使用OpenAI进行综述。'
    )
    
    parser.add_argument('url', type=str, help='要下载的视频的URL')
    parser.add_argument('--cookies', type=str, default="cookies.txt", help='用于yt-dlp的cookies文件名')
    parser.add_argument('--chunk_min', type=int, default=20, help='主块处理时长（分钟）')
    parser.add_argument('--target_sec', type=int, default=60, help='每个切分片段的目标时长（秒）')
    parser.add_argument('--silence_db', type=int, default=-55, help='静音检测的阈值 (dBFS)')
    parser.add_argument('--min_silence_ms', type=int, default=300, help='最小静音长度（毫秒）')
    parser.add_argument('--openai_model', type=str, default="gpt-3.5-turbo", help='用于综述的OpenAI模型')
    parser.add_argument('--skip_summary', action='store_true', help='跳过OpenAI综述步骤')
    parser.add_argument('--force', action='store_true', help='强制执行所有步骤，忽略已存在的文件')
    parser.add_argument('--with-voice', action='store_true', help='使用Fish Audio将综述文本转换为语音')
    parser.add_argument('--fish_reference_id', type=str, help='Fish Audio模型引用ID')
    parser.add_argument('--fish_api_key', type=str, help='Fish Audio API密钥')
    
    args = parser.parse_args()
    
    # 提取视频ID
    video_id = get_video_id_from_url(args.url)
    if not video_id:
        logger.warning(f"无法从URL {args.url} 中提取video_id，使用'unknown_video'作为ID")
        video_id = "unknown_video"
    
    # 创建输出目录
    video_output_dir = create_output_directories(video_id)
    
    # 定义文件路径
    downloaded_mp3_file = os.path.join(video_output_dir, f"{video_id}.mp3")
    result_file = os.path.join(video_output_dir, "result.txt")
    summary_file = os.path.join(video_output_dir, "result_summary.txt")
    voice_file = os.path.join(video_output_dir, "summary_voice.mp3")
    
    # 显示系统信息
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            logger.warning(f"获取GPU设备名称失败: {e}")
    
    # 检查cookies文件
    if not os.path.exists(args.cookies):
        logger.warning(f"Cookies文件 '{args.cookies}' 未找到，下载可能会失败或受限")
    
    # 步骤1: 下载音频
    download_needed = check_file_exists_and_force(downloaded_mp3_file, "音频文件", args.force)
    download_success = True
    
    if download_needed:
        download_success = download_audio(
            args.url, 
            output_path=downloaded_mp3_file, 
            cookies_file=args.cookies
        )
    
    if not download_success:
        logger.error("音频下载失败，无法继续处理")
        return 1
    
    if not os.path.exists(downloaded_mp3_file):
        logger.error(f"下载的音频文件 {downloaded_mp3_file} 未找到")
        return 1
    
    # 步骤2: 音频切分和转录
    transcribe_needed = check_file_exists_and_force(result_file, "转录结果文件", args.force)
    
    if transcribe_needed:
        logger.info(f"开始处理音频文件: {downloaded_mp3_file}")
        result_file_path = smart_split_mp3(
            downloaded_mp3_file,
            output_dir=video_output_dir,
            target_duration_sec=args.target_sec,
            silence_thresh=args.silence_db,
            min_silence_len=args.min_silence_ms,
            chunk_duration_min=args.chunk_min
        )
        
        if not result_file_path:
            logger.error("音频切分和转录失败")
            return 1
    
    # 步骤3: OpenAI综述
    if not args.skip_summary:
        summary_needed = check_file_exists_and_force(summary_file, "综述结果文件", args.force)
        
        if summary_needed and os.path.exists(result_file):
            summary_result = summarize_text_with_openai(
                result_file, 
                output_dir=video_output_dir, 
                model=args.openai_model
            )
            
            if not summary_result:
                logger.warning("综述生成失败")
        elif not os.path.exists(result_file):
            logger.error(f"转录结果文件 {result_file} 未找到，无法进行综述")
    else:
        logger.info("已跳过OpenAI综述步骤")
    
    # 步骤4: 语音生成（可选）
    if getattr(args, 'with_voice', False):
        if not FISHAUDIO_AVAILABLE:
            logger.error("Fish Audio SDK未安装，无法生成语音")
        elif os.path.exists(summary_file):
            voice_needed = check_file_exists_and_force(voice_file, "语音文件", args.force)
            
            if voice_needed:
                voice_result = generate_voice_from_summary(
                    summary_file,
                    output_dir=video_output_dir,
                    reference_id=args.fish_reference_id,
                    api_key=args.fish_api_key
                )
                
                if not voice_result:
                    logger.warning("语音生成失败")
        else:
            logger.error(f"综述文件 {summary_file} 未找到，无法生成语音")
    
    logger.info("所有处理步骤完成")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行过程中发生未预期的错误: {e}")
        sys.exit(1)