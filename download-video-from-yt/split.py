from pydub import AudioSegment
from pydub.silence import detect_silence
import math
import platform
import os
import subprocess # 用于调用ffmpeg 和 yt-dlp
import tempfile # 用于创建临时文件
from mutagen.mp3 import MP3
import argparse # 用于解析命令行参数
import torch # 移到顶部，以便尽早检查CUDA
from openai import OpenAI # 导入OpenAI库
from dotenv import load_dotenv
# 从.env文件加载环境变量
load_dotenv()
def get_audio_duration_fast(file_path):
    """
    快速获取音频文件时长，不加载文件内容
    """
    try:
        audio_file = MP3(file_path)
        return int(audio_file.info.length * 1000)  # 转换为毫秒
    except Exception as e:
        print(f"使用mutagen获取时长失败: {e}")
        return get_duration_with_ffprobe(file_path)

def get_duration_with_ffprobe(file_path):
    """
    使用ffprobe获取音频时长
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_sec = float(result.stdout.strip())
        return int(duration_sec * 1000)  # 转换为毫秒
    except Exception as e:
        print(f"使用ffprobe获取时长失败: {e}")
        return estimate_duration_by_filesize(file_path)

def estimate_duration_by_filesize(file_path):
    """
    根据文件大小估算时长（不够精确，仅作备用）
    """
    file_size = os.path.getsize(file_path)
    estimated_duration_sec = file_size / (128 * 1024 / 8) # 假设平均比特率为128kbps
    return int(estimated_duration_sec * 1000)

def extract_chunk_with_ffmpeg(input_path, output_path, start_ms, end_ms):
    """
    使用ffmpeg提取音频片段
    """
    start_sec = start_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ss', str(start_sec),
        '-t', str(duration_sec),
        '-c', 'copy', # 直接复制流，速度快
        output_path, '-y' # 覆盖输出文件
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg提取失败: {e}")
        print(f"ffmpeg stderr: {e.stderr.decode()}")
        return False

def smart_split_mp3(file_path, target_duration_sec=60, silence_thresh=-55, min_silence_len=300, seek_window_sec=10, chunk_duration_min=20):
    """
    智能切分MP3文件，力求在停顿处切分，每段大约为指定时长。
    终极优化版本：使用ffmpeg提取块，快速获取文件时长，每次只处理指定时长的文件内容。
    新增功能：自动检测第一个分段的语言，后续分段使用该语言，并将所有转录文本写入result.txt。
    """
    print("快速获取文件信息...")
    total_length_ms = get_audio_duration_fast(file_path)
    if total_length_ms == 0:
        print("无法获取音频时长，退出。")
        return

    print(f"文件总时长: {total_length_ms/1000/60:.2f} 分钟")

    system = platform.system()
    model = None
    detected_language = None  # 用于存储检测到的语言
    output_text_file = "result.txt" # 输出文本文件名

    # 清空或准备result.txt文件
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write("") # 创建或清空文件

    if system == 'Windows':
        import whisper
        # 模型加载推迟到第一次转录前，以便获取GPU信息
    elif system == 'Darwin':
        import mlx_whisper
        # mlx_whisper.transcribe 会自行加载模型
    elif system == 'Linux':
        import whisper
        # 模型加载推迟到第一次转录前
    else:
        raise Exception(f"不支持的操作系统: {system}")

    target_duration_ms = target_duration_sec * 1000
    seek_window_ms = seek_window_sec * 1000
    chunk_duration_ms = chunk_duration_min * 60 * 1000

    global_start_ms = 0
    part_num = 1
    first_transcription_done = False

    with tempfile.TemporaryDirectory() as temp_dir:
        while global_start_ms < total_length_ms:
            chunk_end_ms = min(global_start_ms + chunk_duration_ms, total_length_ms)
            print(f"\n处理主块时间段: {global_start_ms/1000/60:.2f} - {chunk_end_ms/1000/60:.2f} 分钟")

            temp_chunk_path = os.path.join(temp_dir, f"temp_chunk_{part_num}.mp3")
            print(f"使用ffmpeg提取主块到: {temp_chunk_path}")
            if not extract_chunk_with_ffmpeg(file_path, temp_chunk_path, global_start_ms, chunk_end_ms):
                print("无法提取主块，跳过此块。")
                global_start_ms = chunk_end_ms
                continue
            
            print("加载提取的主块...")
            try:
                current_chunk_audio = AudioSegment.from_mp3(temp_chunk_path)
            except Exception as e:
                print(f"加载提取的主块失败: {e}, 跳过此块。")
                os.remove(temp_chunk_path) # 确保删除失败加载的临时文件
                global_start_ms = chunk_end_ms
                continue
            
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path) # 删除临时主块文件

            print("检测静音段...")
            silences = detect_silence(current_chunk_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

            chunk_internal_start_ms = 0
            while chunk_internal_start_ms < len(current_chunk_audio):
                chunk_internal_end_candidate = min(chunk_internal_start_ms + target_duration_ms, len(current_chunk_audio))
                search_start = max(0, chunk_internal_end_candidate - seek_window_ms)
                search_end = min(len(current_chunk_audio), chunk_internal_end_candidate + seek_window_ms)
                split_point = chunk_internal_end_candidate
                best_silence_start = None
                min_diff = float('inf')

                for silence_start_loop, silence_end_loop in silences: # Renamed to avoid conflict
                    if search_start <= silence_start_loop <= search_end:
                        if silence_start_loop >= chunk_internal_end_candidate:
                            diff = silence_start_loop - chunk_internal_end_candidate
                            if diff < min_diff:
                                min_diff = diff
                                best_silence_start = silence_start_loop
                                break # Found a good point after candidate
                        diff = abs(silence_start_loop - chunk_internal_end_candidate)
                        if diff < min_diff:
                            min_diff = diff
                            best_silence_start = silence_start_loop
                
                if best_silence_start is not None:
                    split_point = best_silence_start
                
                if split_point <= chunk_internal_start_ms: # Ensure progress
                    split_point = min(chunk_internal_start_ms + target_duration_ms, len(current_chunk_audio))
                
                # Ensure split_point does not exceed current_chunk_audio length
                split_point = min(split_point, len(current_chunk_audio))

                if chunk_internal_start_ms >= split_point: # Avoid creating empty segments if stuck
                    if chunk_internal_start_ms < len(current_chunk_audio):
                         split_point = len(current_chunk_audio) # Process the rest of the chunk
                    else:
                        break # No more audio in this chunk

                segment = current_chunk_audio[chunk_internal_start_ms:split_point]
                if len(segment) == 0: # Skip empty segments
                    chunk_internal_start_ms = split_point
                    if split_point >= len(current_chunk_audio):
                        break
                    continue

                output_filename = f"{os.path.splitext(file_path)[0]}_part_{part_num:03d}.mp3"
                segment.export(output_filename, format="mp3")
                print(f"导出 {output_filename} (时长: {len(segment)/1000:.2f} 秒)")

                try:
                    transcribe_language = detected_language if detected_language else None # Use None for auto-detect on first pass
                    initial_prompt_text = "transcribe the following video." # Default prompt
                    if detected_language == "zh" or (transcribe_language is None and not first_transcription_done): # Special prompt for Chinese or first detection
                        initial_prompt_text = "以下是普通话的句子，请转写成简体中文。"
                    
                    if model is None and (system == 'Windows' or system == 'Linux'):
                        print("加载Whisper模型...")
                        model = whisper.load_model("large-v3") # Changed to large-v3 as turbo might not be standard
                    
                    if system == 'Windows' or system == 'Linux':
                        result = model.transcribe(output_filename, language=transcribe_language, initial_prompt=initial_prompt_text)
                    elif system == 'Darwin':
                        # For mlx_whisper, language detection is usually part of the transcribe call if language is None
                        # We'll set it after the first transcription
                        result = mlx_whisper.transcribe(output_filename, 
                                                      language=transcribe_language, 
                                                      initial_prompt=initial_prompt_text, 
                                                      path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
                    
                    if not first_transcription_done:
                        detected_language = result.get('language', 'en') # Default to 'en' if not detected
                        print(f"检测到的语言: {detected_language}")
                        first_transcription_done = True
                        # Update prompt if language is now known and it's Chinese
                        if detected_language == "zh":
                            initial_prompt_text = "以下是普通话的句子，请转写成简体中文。"

                    print(f"转录结果 ({detected_language}): {result['text']}")
                    with open(output_text_file, 'a', encoding='utf-8') as f:
                        f.write(result['text'] + '\n')

                except Exception as e:
                    print(f"转录出错: {e}")
                
                del segment
                chunk_internal_start_ms = split_point
                part_num += 1
                if split_point >= len(current_chunk_audio):
                    break
            
            del current_chunk_audio
            global_start_ms = chunk_end_ms
            print(f"完成处理主块: {global_start_ms/1000/60:.2f} - {chunk_end_ms/1000/60:.2f} 分钟")

    # (保持 smart_split_mp3 函数内容不变，因为其逻辑是复用的)
    # ... 确保之前的语言检测和文件写入逻辑仍然存在 ...
    print(f"\n所有切分完成！转录结果已保存到 {output_text_file}")
    return output_text_file # 返回result.txt的文件路径，方便后续处理

def download_audio(url, output_filename="input.mp3", cookies_file="cookies.txt"):
    """
    使用 yt-dlp 下载音频。
    """
    print(f"开始下载音频从 URL: {url}")
    command = [
        'yt-dlp',
        '--cookies', cookies_file,
        '--audio-format', 'mp3',
        '-x', # 提取音频
        '-o', output_filename,
        url
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print(f"音频成功下载并保存为: {output_filename}")
            return True
        else:
            print(f"下载失败。错误码: {process.returncode}")
            print(f"yt-dlp stdout: {stdout.decode(errors='ignore')}")
            print(f"yt-dlp stderr: {stderr.decode(errors='ignore')}")
            return False
    except FileNotFoundError:
        print("错误: yt-dlp 命令未找到。请确保它已安装并配置在系统PATH中。")
        return False
    except Exception as e:
        print(f"下载过程中发生错误: {e}")
        return False

def summarize_text_with_openai(text_file_path, model="qwen/qwen3-235b-a22b:free"):
    """
    使用OpenAI API对指定文本文件的内容进行综述。
    """
    print(f"\n开始使用OpenAI对 {text_file_path} 的内容进行综述...")
    try:
        # 确保 API密钥已设置 (通常通过环境变量 OPENAI_API_KEY)
        if not os.getenv("OPENAI_API_KEY"):
            print("错误: OPENAI_API_KEY 环境变量未设置。")
            print("请设置您的OpenAI API密钥后重试。")
            return None

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量获取API密钥
            base_url='https://openrouter.ai/api/v1'
        )

        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print("文本文件内容为空，无法进行综述。")
            return None

        # 构建prompt
        # 您可以根据需要调整这个prompt
        prompt_message = f"请将以下内容转换成流畅易懂的中文口语化的口播稿，内容要清晰且连贯：\n\n{content}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_message}
            ]
        )
        
        summary = response.choices[0].message.content
        print("\n--- OpenAI 综述结果 ---")
        print(summary)
        print("--- 综述结束 ---")
        
        
        # 可以选择将综述结果保存到文件
        summary_file_path = f"{os.path.splitext(text_file_path)[0]}_summary.txt"
        with open(summary_file_path, 'w', encoding='utf-8') as f_summary:
            f_summary.write(summary)
        print(f"综述结果已保存到: {summary_file_path}")
        return summary

    except ImportError:
        print("错误: openai 库未安装。请运行 'pip install openai' 安装。")
        return None
    except Exception as e:
        print(f"调用OpenAI API时发生错误: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载YouTube视频的音频，进行智能切分和转录，并使用OpenAI进行综述。')
    parser.add_argument('url', type=str, help='要下载的视频的URL。')
    parser.add_argument('--output_mp3', type=str, default="input.mp3", help='下载的MP3文件名。')
    parser.add_argument('--cookies', type=str, default="cookies.txt", help='用于yt-dlp的cookies文件名。')
    parser.add_argument('--chunk_min', type=int, default=20, help='smart_split_mp3处理的主块时长（分钟）。')
    parser.add_argument('--target_sec', type=int, default=60, help='每个切分片段的目标时长（秒）。')
    parser.add_argument('--silence_db', type=int, default=-55, help='静音检测的阈值 (dBFS)。')
    parser.add_argument('--min_silence_ms', type=int, default=300, help='最小静音长度（毫秒）。')
    parser.add_argument('--openai_model', type=str, default="gpt-3.5-turbo", help='用于综述的OpenAI模型。')
    parser.add_argument('--skip_summary', action='store_true', help='如果设置，则跳过OpenAI综述步骤。')

    args = parser.parse_args()

    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"获取GPU设备名称失败: {e}")

    if not os.path.exists(args.cookies):
        print(f"警告: Cookies 文件 '{args.cookies}' 未找到。下载可能会失败或受限。")

    if download_audio(args.url, output_filename=args.output_mp3, cookies_file=args.cookies):
        if os.path.exists(args.output_mp3):
            print(f"\n开始处理音频文件: {args.output_mp3}")
            result_file = smart_split_mp3(
                args.output_mp3, 
                target_duration_sec=args.target_sec,
                silence_thresh=args.silence_db,
                min_silence_len=args.min_silence_ms,
                chunk_duration_min=args.chunk_min
            )

            if result_file and os.path.exists(result_file) and not args.skip_summary:
                summarize_text_with_openai(result_file, model=args.openai_model)
            elif args.skip_summary:
                print("已跳过OpenAI综述步骤。")
            elif not result_file or not os.path.exists(result_file):
                print(f"错误: 转录结果文件 {result_file if result_file else 'result.txt'} 未找到，无法进行综述。")
        else:
            print(f"错误: 下载的音频文件 {args.output_mp3} 未找到，尽管下载报告成功。")
    else:
        print("音频下载失败，无法继续处理。")