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
import re
from urllib.parse import urlparse, parse_qs  # 添加URL解析相关的库
from tqdm import tqdm  # 导入进度条库
import time  # 用于模拟进度
import sys  # 用于实时输出
import logging  # 导入日志模块

# 添加CosyVoice相关导入
try:
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice
    COSYVOICE_AVAILABLE = True
except ImportError as e:
    print({e})
    COSYVOICE_AVAILABLE = False
    print("警告: CosyVoice相关库未安装，--with-voice选项将不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("running_log_for_transcribe.log"),
        logging.StreamHandler()
    ]
)

# 创建logger对象
logger = logging.getLogger('running_log_for_transcribe')

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
        logger.error(f"使用mutagen获取时长失败: {e}")
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
        logger.error(f"使用ffprobe获取时长失败: {e}")
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

def get_video_id_from_url(url):
    """
    从YouTube URL中提取video_id。
    """
    try:
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.hostname or 'youtu.be' in parsed_url.hostname:
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                return query_params.get('v', [None])[0]
            elif parsed_url.path.startswith('/'): # For youtu.be/VIDEO_ID format
                return parsed_url.path[1:]
    except Exception as e:
        print(f"解析URL提取video_id失败: {e}")
    # 备用方案：简单的正则匹配，可能不够健壮
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    return None

def get_audio_duration_fast(file_path):
    """
    快速获取音频文件时长，不加载文件内容
    """
    try:
        audio_file = MP3(file_path)
        return int(audio_file.info.length * 1000)  # 转换为毫秒
    except Exception as e:
        logger.error(f"使用mutagen获取时长失败: {e}")
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
        logger.error(f"使用ffprobe获取时长失败: {e}")
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

def smart_split_mp3(file_path, output_dir, target_duration_sec=60, silence_thresh=-55, min_silence_len=300, seek_window_sec=10, chunk_duration_min=20):
    """
    智能切分MP3文件，并将所有输出（分段、转录文本）保存到指定的output_dir。
    """
    total_length_ms = get_audio_duration_fast(file_path)
    if total_length_ms == 0:
        logger.error("无法获取音频时长，退出。")
        return None

    logger.info(f"文件总时长: {total_length_ms/1000} 秒")

    system = platform.system()
    model = None
    detected_language = None
    # 输出文本文件名现在基于output_dir
    output_text_file = os.path.join(output_dir, "result.txt")


    if system == 'Windows':
        import whisper
    elif system == 'Darwin':
        import mlx_whisper
    elif system == 'Linux':
        import whisper
    else:
        raise Exception(f"不支持的操作系统: {system}")

    target_duration_ms = target_duration_sec * 1000
    seek_window_ms = seek_window_sec * 1000
    chunk_duration_ms = chunk_duration_min * 60 * 1000

    global_start_ms = 0
    part_num = 1
    first_transcription_done = False

    # 创建总进度条
    total_progress = tqdm(total=total_length_ms/1000, desc="总体进度", unit="ms", unit_scale=True, position=0)
    
    # 主临时目录仍然使用tempfile，但分段MP3会保存到output_dir
    with tempfile.TemporaryDirectory() as temp_dir_for_chunks:
        while global_start_ms < total_length_ms:
            chunk_end_ms = min(global_start_ms + chunk_duration_ms, total_length_ms)
            #print(f"\n处理主块时间段: {global_start_ms/1000/60:.2f} - {chunk_end_ms/1000/60:.2f} 分钟")

            # 临时主块文件仍在临时目录中
            temp_main_chunk_path = os.path.join(temp_dir_for_chunks, f"temp_main_chunk_{part_num}.mp3")
            
            # 提取主块进度条
            extract_desc = f"提取主块 {part_num}"

            success = extract_chunk_with_ffmpeg(file_path, temp_main_chunk_path, global_start_ms, chunk_end_ms)
                
            if not success:
                print("无法提取主块，跳过此块。")
                global_start_ms = chunk_end_ms
                continue
            
            #print("加载提取的主块...")
            try:
                current_chunk_audio = AudioSegment.from_mp3(temp_main_chunk_path)
            except Exception as e:
                print(f"加载提取的主块失败: {e}, 跳过此块。")
                if os.path.exists(temp_main_chunk_path):
                    os.remove(temp_main_chunk_path)
                global_start_ms = chunk_end_ms
                
                continue
            
            if os.path.exists(temp_main_chunk_path):
                os.remove(temp_main_chunk_path)

            #print("检测静音段...")
            silences = detect_silence(current_chunk_audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

            # 创建分段进度条
            #segment_progress = tqdm(total=len(current_chunk_audio), desc="分段进度", unit="ms", position=1, leave=False)
            
            chunk_internal_start_ms = 0
            while chunk_internal_start_ms < len(current_chunk_audio):
                chunk_internal_end_candidate = min(chunk_internal_start_ms + target_duration_ms, len(current_chunk_audio))
                search_start = max(0, chunk_internal_end_candidate - seek_window_ms)
                search_end = min(len(current_chunk_audio), chunk_internal_end_candidate + seek_window_ms)
                split_point = chunk_internal_end_candidate
                best_silence_start = None
                min_diff = float('inf')

                for silence_start_loop, silence_end_loop in silences: 
                    if search_start <= silence_start_loop <= search_end:
                        if silence_start_loop >= chunk_internal_end_candidate:
                            diff = silence_start_loop - chunk_internal_end_candidate
                            if diff < min_diff:
                                min_diff = diff
                                best_silence_start = silence_start_loop
                                break 
                        diff = abs(silence_start_loop - chunk_internal_end_candidate)
                        if diff < min_diff:
                            min_diff = diff
                            best_silence_start = silence_start_loop
                
                if best_silence_start is not None:
                    split_point = best_silence_start
                
                if split_point <= chunk_internal_start_ms: 
                    split_point = min(chunk_internal_start_ms + target_duration_ms, len(current_chunk_audio))
                
                split_point = min(split_point, len(current_chunk_audio))

                if chunk_internal_start_ms >= split_point: 
                    if chunk_internal_start_ms < len(current_chunk_audio):
                         split_point = len(current_chunk_audio) 
                    else:
                        break 

                segment = current_chunk_audio[chunk_internal_start_ms:split_point]
                if len(segment) == 0: 
                    chunk_internal_start_ms = split_point
                    if split_point >= len(current_chunk_audio):
                        break
                    continue
                
                # 输出分段MP3到指定的output_dir
                base_mp3_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_segment_filename = os.path.join(output_dir, f"{base_mp3_filename}_part_{part_num:03d}.mp3")
                segment.export(output_segment_filename, format="mp3")
                #print(f"导出 {output_segment_filename} (时长: {len(segment)/1000:.2f} 秒)")

                try:
                    transcribe_language = detected_language if detected_language else None
                    initial_prompt_text = "transcribe the following audio."
                    if detected_language == "zh" or (transcribe_language is None and not first_transcription_done):
                        initial_prompt_text = "提取音频中的文字内容。"
                    
                    if model is None and (system == 'Windows' or system == 'Linux'):
                        #print("加载Whisper模型...")
                        model = whisper.load_model("large-v3")
                    
                    if system == 'Windows' or system == 'Linux':
                        result = model.transcribe(output_segment_filename, language=transcribe_language, initial_prompt=initial_prompt_text)
                    elif system == 'Darwin':
                        result = mlx_whisper.transcribe(output_segment_filename, 
                                                      language=transcribe_language, 
                                                      initial_prompt=initial_prompt_text, 
                                                      path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
                    

                    # print(f"转录结果 ({detected_language}): {result['text']}")
                    with open(output_text_file, 'a', encoding='utf-8') as f_text:
                        f_text.write(result['text'] + '\n')
                    total_progress.update(len(segment)/1000)
                except Exception as e:
                    print(f"转录 {output_segment_filename} 出错: {e}")
                
                del segment
                chunk_internal_start_ms = split_point
                part_num += 1
                if split_point >= len(current_chunk_audio):
                    break
                os.remove(output_segment_filename)
            del current_chunk_audio
            global_start_ms = chunk_end_ms
            #print(f"完成处理主块: {global_start_ms/1000/60:.2f} - {chunk_end_ms/1000/60:.2f} 分钟")


    # 关闭总进度条
    total_progress.close()
    print(f"\n所有切分完成！转录结果已保存到 {output_text_file}")
    return output_text_file

def download_audio(url, output_path, cookies_file="cookies.txt"):
    """
    使用 yt-dlp 下载音频到指定的output_path。
    """
    logger.info(f"开始下载音频从 URL: {url} 到 {output_path}")
    command = [
        'yt-dlp',
        '--cookies', cookies_file,
        '--audio-format', 'mp3',
        '-x', 
        '-o', output_path, # 直接使用完整的输出路径
        url
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            logger.info(f"音频成功下载并保存为: {output_path}")
            return True
        else:
            logger.error(f"下载失败。错误码: {process.returncode}")
            logger.debug(f"yt-dlp stdout: {stdout.decode(errors='ignore')}")
            logger.debug(f"yt-dlp stderr: {stderr.decode(errors='ignore')}")
            return False
    except FileNotFoundError:
        logger.error("错误: yt-dlp 命令未找到。请确保它已安装并配置在系统PATH中。")
        return False
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        return False

def summarize_text_with_openai(text_file_path, output_dir, model="google/gemini-2.0-flash-exp:free"):
    """
    使用OpenAI API对指定文本文件的内容进行综述，并将综述结果保存到output_dir。
    """
    logger.info(f"开始使用OpenAI对 {text_file_path} 的内容进行综述...")
    try:
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("错误: OPENAI_API_KEY 环境变量未设置。")
            return None
        if not os.getenv("BASE_URL"):
            logger.error("错误: BASE_URL 环境变量未设置。")
            return None

        # 优先使用环境变量中的MODEL_NAME，如果未设置则使用传入的model参数
        env_model = os.getenv("MODEL_NAME")
        if env_model and env_model.strip():
            model = env_model
            logger.info(f"使用环境变量中的模型: {model}")
        else:
            logger.info(f"使用默认模型: {model}")
        
        # 创建OpenAI客户端实例，使用环境变量中的API密钥和基础URL
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        with open(text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            print("文本文件内容为空，无法进行综述。")
            return None

        prompt_message = f'''
        {content}
        以上是一份优秀的口播稿件。
        你是一个优秀的口播撰稿人，你写稿子的特点如下：
        1、你的频道的名字是kaka乱弹。一般以“hello，欢迎来到kaka乱弹的频道”开头。
        2、你面向的听众是年轻白领，你的性格是年轻、爱美、活泼、聪明。
        3、你善于把握重点，擅长对复杂的问题进行形象化举例说明。
        4、你的语言丰富，擅长用网络梗，不喜欢用markdown、表情包、代码块写稿
        5、你的语言风格接地气，段落和内容切换自然合理，你会规避使用1、2、3.。。这样的罗列信息的方式来表达，而是采用自然、口语化的方式来衔接
        6、你学习前面给出的内容，但会避免使用内容中具有个人风格的表达方式，避免使用内容中带有的广告、宣传推广的内容。
        现在，请你请学习上面的知识，按照你自己的角色风格重新写一份精彩的口播演讲稿。'''
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
        
        # 综述文件保存到output_dir
        summary_file_name = f"{os.path.splitext(os.path.basename(text_file_path))[0]}_summary.txt"
        summary_file_path = os.path.join(output_dir, summary_file_name)
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

def split_text_into_sentences(text):
    """
    将文本按句子拆分，支持中英文句号、问号、感叹号
    """
    # 使用正则表达式按句子分割，支持中英文标点
    sentences = re.split(r'[。！？.!?]+', text)
    # 过滤空字符串并去除首尾空格
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_voice_from_summary(summary_file_path, output_dir, model_path="pretrained_models/CosyVoice-300M-SFT", speaker="中文女"):
    """
    使用CosyVoice将综述文本转换为语音
    """
    if not COSYVOICE_AVAILABLE:
        logger.error("CosyVoice库未安装，无法生成语音")
        return None
    
    try:
        logger.info(f"开始使用CosyVoice生成语音，模型路径: {model_path}")
        
        # 读取综述文本
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
        
        if not summary_text:
            logger.error("综述文件为空，无法生成语音")
            return None
        
        # 初始化CosyVoice模型
        logger.info("正在加载CosyVoice模型...")
        cosyvoice = CosyVoice(model_path, load_jit=False, load_trt=False, fp16=False)
        
        # 将文本按句子拆分
        sentences = split_text_into_sentences(summary_text)
        logger.info(f"文本已拆分为 {len(sentences)} 个句子")
        
        # 生成每个句子的音频
        audio_segments = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            logger.info(f"正在生成第 {i+1}/{len(sentences)} 个句子的语音: {sentence[:50]}...")
            
            try:
                # 使用SFT模式生成语音
                for j, result in enumerate(cosyvoice.inference_sft(sentence, speaker, stream=False)):
                    audio_segments.append(result['tts_speech'])
                    break  # 只取第一个结果
            except Exception as e:
                logger.error(f"生成第 {i+1} 个句子的语音时出错: {e}")
                continue
        
        if not audio_segments:
            logger.error("没有成功生成任何音频片段")
            return None
        
        # 合并所有音频片段
        logger.info("正在合并音频片段...")
        combined_audio = torch.cat(audio_segments, dim=1)
        
        # 保存合并后的音频文件
        output_audio_file = os.path.join(output_dir, "summary_voice.wav")
        torchaudio.save(output_audio_file, combined_audio, cosyvoice.sample_rate)
        
        logger.info(f"语音文件已保存到: {output_audio_file}")
        print(f"语音文件已生成: {output_audio_file}")
        
        return output_audio_file
        
    except Exception as e:
        logger.error(f"生成语音时发生错误: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载YouTube视频的音频，进行智能切分和转录，并使用OpenAI进行综述。所有输出保存到 output/{video_id}/ 目录。')
    parser.add_argument('url', type=str, help='要下载的视频的URL。')
    # 移除了 --output_mp3 参数，因为它将基于video_id动态生成
    parser.add_argument('--cookies', type=str, default="cookies.txt", help='用于yt-dlp的cookies文件名。')
    parser.add_argument('--chunk_min', type=int, default=20, help='smart_split_mp3处理的主块时长（分钟）。')
    parser.add_argument('--target_sec', type=int, default=60, help='每个切分片段的目标时长（秒）。')
    parser.add_argument('--silence_db', type=int, default=-55, help='静音检测的阈值 (dBFS)。')
    parser.add_argument('--min_silence_ms', type=int, default=300, help='最小静音长度（毫秒）。')
    parser.add_argument('--openai_model', type=str, default="gpt-3.5-turbo", help='用于综述的OpenAI模型。')
    parser.add_argument('--skip_summary', action='store_true', help='如果设置，则跳过OpenAI综述步骤。')
    parser.add_argument('--force', action='store_true', help='强制执行所有步骤，忽略已下载文件和已生成文本。')
    parser.add_argument('--with-voice', action='store_true', help='如果设置，则使用CosyVoice将综述文本转换为语音。')
    parser.add_argument('--cosyvoice_model', type=str, default="pretrained_models/CosyVoice-300M-SFT", help='CosyVoice模型路径。')
    parser.add_argument('--voice_speaker', type=str, default="中文女", help='CosyVoice语音说话人。')

    args = parser.parse_args()

    video_id = get_video_id_from_url(args.url)
    if not video_id:
        logger.warning(f"无法从URL {args.url} 中提取 video_id。将使用 'unknown_video' 作为ID。")
        video_id = "unknown_video"
    
    # 创建主输出目录 output/
    main_output_dir = "output"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
        logger.debug(f"创建主输出目录: {main_output_dir}")

    # 创建特定视频的输出目录 output/{video_id}/
    video_specific_output_dir = os.path.join(main_output_dir, video_id)
    if not os.path.exists(video_specific_output_dir):
        os.makedirs(video_specific_output_dir)
        logger.debug(f"创建视频专属输出目录: {video_specific_output_dir}")

    # 定义下载的MP3文件名，存放在视频专属目录中
    downloaded_mp3_filename = os.path.join(video_specific_output_dir, f"{video_id}.mp3")
    # 定义转录结果文件路径
    result_file = os.path.join(video_specific_output_dir, "result.txt")
    # 定义综述结果文件路径
    summary_file = os.path.join(video_specific_output_dir, "result_summary.txt")
    # 定义语音文件路径
    voice_file = os.path.join(video_specific_output_dir, "summary_voice.wav")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"获取GPU设备名称失败: {e}")

    if not os.path.exists(args.cookies):
        print(f"警告: Cookies 文件 '{args.cookies}' 未找到。下载可能会失败或受限。")

    # 检查音频文件是否已下载
    download_needed = True
    if os.path.exists(downloaded_mp3_filename) and not args.force:
        print(f"音频文件已存在: {downloaded_mp3_filename}")
        print("跳过下载步骤。如需重新下载，请使用 --force 选项。")
        download_needed = False
    
    # 下载音频文件（如果需要）
    download_success = True
    if download_needed:
        download_success = download_audio(args.url, output_path=downloaded_mp3_filename, cookies_file=args.cookies)
    
    if download_success and os.path.exists(downloaded_mp3_filename):
        # 检查转录结果是否已生成
        transcribe_needed = True
        if os.path.exists(result_file) and not args.force:
            print(f"转录结果文件已存在: {result_file}")
            print("跳过文本提取步骤。如需重新提取，请使用 --force 选项。")
            transcribe_needed = False
        
        # 提取文本（如果需要）
        if transcribe_needed:
            print(f"\n开始处理音频文件: {downloaded_mp3_filename}")
            result_file = smart_split_mp3(
                downloaded_mp3_filename, 
                output_dir=video_specific_output_dir,
                target_duration_sec=args.target_sec,
                silence_thresh=args.silence_db,
                min_silence_len=args.min_silence_ms,
                chunk_duration_min=args.chunk_min
            )
        
        # 检查是否需要进行综述
        summary_needed = not args.skip_summary
        if os.path.exists(summary_file) and not args.force and summary_needed:
            print(f"综述结果文件已存在: {summary_file}")
            print("跳过综述步骤。如需重新生成综述，请使用 --force 选项。")
            summary_needed = False
        
        # 进行综述（如果需要）
        if summary_needed and os.path.exists(result_file):
            summarize_text_with_openai(result_file, output_dir=video_specific_output_dir, model=args.openai_model)
        elif args.skip_summary:
            print("已跳过OpenAI综述步骤。")
        elif not os.path.exists(result_file):
            print(f"错误: 转录结果文件 {result_file} 未找到，无法进行综述。")
        
        # 生成语音（如果需要且综述文件存在）
        if getattr(args, 'with_voice', False):  # 使用getattr处理连字符参数
            if not COSYVOICE_AVAILABLE:
                print("错误: CosyVoice库未安装，无法生成语音。请安装CosyVoice相关依赖。")
            elif os.path.exists(summary_file):
                # 检查语音文件是否已生成
                voice_needed = True
                if os.path.exists(voice_file) and not args.force:
                    print(f"语音文件已存在: {voice_file}")
                    print("跳过语音生成步骤。如需重新生成，请使用 --force 选项。")
                    voice_needed = False
                
                if voice_needed:
                    generate_voice_from_summary(
                        summary_file, 
                        output_dir=video_specific_output_dir,
                        model_path=args.cosyvoice_model,
                        speaker=args.voice_speaker
                    )
            else:
                print(f"错误: 综述文件 {summary_file} 未找到，无法生成语音。请先生成综述。")

if not download_success:
    print("音频下载失败，无法继续处理。")
elif not os.path.exists(downloaded_mp3_filename):
    print(f"错误: 下载的音频文件 {downloaded_mp3_filename} 未找到，尽管下载报告成功。")