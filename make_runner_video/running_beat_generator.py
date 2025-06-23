import numpy as np
import scipy.io.wavfile as wav
import math
import argparse

def generate_metronome_beat(bpm=170, duration_minutes=10, sample_rate=44100):
    """
    生成节拍器音频
    
    参数:
    bpm: 每分钟节拍数 (默认170)
    duration_minutes: 音频时长(分钟)
    sample_rate: 采样率
    """
    
    # 计算参数
    beat_interval = 60.0 / bpm  # 每个节拍的间隔(秒)
    total_samples = int(sample_rate * duration_minutes * 60)
    audio = np.zeros(total_samples)
    
    # 节拍音参数
    beat_frequency = 800  # 节拍音频率(Hz)
    beat_duration = 0.1   # 每个节拍音的持续时间(秒)
    beat_samples = int(sample_rate * beat_duration)
    
    # 生成节拍音波形
    t_beat = np.linspace(0, beat_duration, beat_samples)
    beat_wave = np.sin(2 * np.pi * beat_frequency * t_beat)
    
    # 添加包络以避免爆音
    envelope = np.exp(-t_beat * 10)  # 指数衰减包络
    beat_wave *= envelope
    
    # 在音频中添加节拍
    current_time = 0
    while current_time < duration_minutes * 60:
        start_sample = int(current_time * sample_rate)
        end_sample = start_sample + beat_samples
        
        if end_sample <= total_samples:
            audio[start_sample:end_sample] += beat_wave
        
        current_time += beat_interval
    
    # 归一化音频
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate

def generate_comfortable_running_music(bpm=170, duration_minutes=10):
    """
    生成舒适的慢跑音乐节拍
    
    参数:
    bpm: 每分钟节拍数
    duration_minutes: 音频时长(分钟)
    """
    
    sample_rate = 44100
    beat_interval = 60.0 / bpm
    total_samples = int(sample_rate * duration_minutes * 60)
    audio = np.zeros(total_samples)
    
    # 主节拍 - 低频鼓声
    def create_kick_drum(duration=0.15, frequency=60):
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 使用正弦波模拟鼓声，添加频率扫描效果
        freq_sweep = frequency * (1 + 2 * np.exp(-t * 20))
        kick = np.sin(2 * np.pi * freq_sweep * t)
        # 添加指数衰减包络
        envelope = np.exp(-t * 8)
        return kick * envelope
    
    # 辅助节拍 - 高频点击声
    def create_hi_hat(duration=0.05, frequency=8000):
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 使用白噪声模拟镲片声
        noise = np.random.normal(0, 0.1, len(t))
        # 高通滤波效果
        hi_hat = noise * np.sin(2 * np.pi * frequency * t)
        # 快速衰减
        envelope = np.exp(-t * 30)
        return hi_hat * envelope
    
    # 生成节拍模式
    kick_sound = create_kick_drum()
    hi_hat_sound = create_hi_hat()
    
    # 添加节拍到音频中
    beat_count = 0
    current_time = 0
    
    while current_time < duration_minutes * 60:
        start_sample = int(current_time * sample_rate)
        
        # 每4拍一个循环，第1和第3拍是重音
        if beat_count % 4 == 0 or beat_count % 4 == 2:
            # 主节拍
            end_sample = start_sample + len(kick_sound)
            if end_sample <= total_samples:
                audio[start_sample:end_sample] += kick_sound * 0.8
        else:
            # 辅助节拍
            end_sample = start_sample + len(hi_hat_sound)
            if end_sample <= total_samples:
                audio[start_sample:end_sample] += hi_hat_sound * 0.4
        
        current_time += beat_interval
        beat_count += 1
    
    # 添加低频背景音调以增加舒适感
    t_total = np.linspace(0, duration_minutes * 60, total_samples)
    background_tone = 0.1 * np.sin(2 * np.pi * 40 * t_total)  # 40Hz低频
    audio += background_tone
    
    # 归一化
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    return audio, sample_rate

def save_audio(audio, sample_rate, filename):
    """
    保存音频文件
    """
    # 转换为16位整数
    audio_int16 = (audio * 32767).astype(np.int16)
    wav.write(filename, sample_rate, audio_int16)
    print(f"音频已保存为: {filename}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成慢跑节拍音乐')
    parser.add_argument('--bpm', type=int, default=170, help='每分钟节拍数 (默认: 170)')
    parser.add_argument('--duration', type=int, default=10, help='音频时长(分钟) (默认: 10)')
    parser.add_argument('--output', type=str, default=None, help='输出文件名 (可选)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    bpm = args.bpm
    duration_minutes = args.duration
    
    print(f"正在生成{bpm}bpm慢跑节拍音乐...")
    
    # 生成音乐
    audio, sample_rate = generate_comfortable_running_music(bpm=bpm, duration_minutes=duration_minutes)
    
    # 确定输出文件名
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"running_beat_{bpm}bpm_{duration_minutes}min.wav"
    
    # 保存文件
    save_audio(audio, sample_rate, output_filename)
    
    print(f"生成完成！")
    print(f"- BPM: {bpm}")
    print(f"- 时长: {duration_minutes}分钟")
    print(f"- 文件: {output_filename}")
    
    # 也可以生成简单的节拍器版本
    print("\n是否生成简单节拍器版本? (y/n): ", end="")
    choice = input().lower()
    if choice == 'y' or choice == 'yes':
        print("正在生成简单节拍器版本...")
        metronome_audio, _ = generate_metronome_beat(bpm=bpm, duration_minutes=duration_minutes)
        metronome_filename = f"metronome_{bpm}bpm_{duration_minutes}min.wav"
        save_audio(metronome_audio, sample_rate, metronome_filename)
        print(f"简单节拍器版本已保存为: {metronome_filename}")

if __name__ == "__main__":
    main()