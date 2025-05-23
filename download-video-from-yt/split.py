from pydub import AudioSegment
from pydub.silence import detect_silence
import math

def smart_split_mp3(file_path, target_duration_sec=60, silence_thresh=-55, min_silence_len=300, seek_window_sec=10):
    """
    智能切分MP3文件，力求在停顿处切分，每段大约为指定时长。

    Args:
        file_path (str): MP3文件的路径。
        target_duration_sec (int): 目标切分时长（秒）。
        silence_thresh (int): 静音阈值（dBFS）。低于此值被认为是静音。
        min_silence_len (int): 最小静音持续时间（毫秒）。
        seek_window_sec (int): 在目标切分点前后查找静音的窗口大小（秒）。
    """
    audio = AudioSegment.from_mp3(file_path)
    total_length_ms = len(audio)
    target_duration_ms = target_duration_sec * 1000
    seek_window_ms = seek_window_sec * 1000

    start_ms = 0
    part_num = 1

    while start_ms < total_length_ms:
        end_ms_candidate = min(start_ms + target_duration_ms, total_length_ms)

        # 定义查找静音的起始和结束点
        search_start_ms = max(start_ms, end_ms_candidate - seek_window_ms)
        search_end_ms = min(total_length_ms, end_ms_candidate + seek_window_ms)

        # 确保 search_start_ms 不会超过 search_end_ms
        if search_start_ms >= search_end_ms:
            # 如果窗口太小或在文件末尾，直接切分到文件结束
            split_point_ms = total_length_ms
        else:
            # 在一个范围内查找静音
            segment_to_search = audio[search_start_ms:search_end_ms]
            silences = detect_silence(segment_to_search, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

            split_point_ms = -1

            if silences:
                # 尝试找到离 target_duration_ms 最近的静音点
                closest_silence_start = -1
                min_diff = float('inf')

                for silence_start, silence_end in silences:
                    # 将静音点转换为原始音频的绝对时间
                    absolute_silence_start = search_start_ms + silence_start
                    
                    # 优先考虑在目标时长之后的静音点，并且距离目标时长越近越好
                    if absolute_silence_start >= end_ms_candidate:
                        diff = abs(absolute_silence_start - end_ms_candidate)
                        if diff < min_diff:
                            min_diff = diff
                            closest_silence_start = absolute_silence_start
                            split_point_ms = closest_silence_start
                            break # 找到第一个符合条件的就用它

                # 如果没有找到目标时长之后的静音点，则找目标时长之前且最近的
                if split_point_ms == -1:
                     min_diff = float('inf')
                     for silence_start, silence_end in silences:
                        absolute_silence_start = search_start_ms + silence_start
                        diff = abs(absolute_silence_start - end_ms_candidate)
                        if diff < min_diff:
                            min_diff = diff
                            closest_silence_start = absolute_silence_start
                            split_point_ms = closest_silence_start
            else:
                # 如果在查找范围内没有找到静音，则在候选结束点切分
                split_point_ms = end_ms_candidate

        # 确保切分点不小于当前起始点
        if split_point_ms <= start_ms:
            split_point_ms = min(start_ms + target_duration_ms, total_length_ms)


        # 切分音频
        chunk = audio[start_ms:split_point_ms]
        output_filename = f"{file_path.split('.')[0]}_part_{part_num:03d}.mp3"
        chunk.export(output_filename, format="mp3")
        print(f"导出 {output_filename} (时长: {len(chunk)/1000:.2f} 秒)")

        start_ms = split_point_ms
        part_num += 1

    print("切分完成！")

if __name__ == "__main__":
    input_mp3_file = "input.mp3"  # 替换为你的MP3文件路径

    # 调用切分函数
    smart_split_mp3(input_mp3_file)