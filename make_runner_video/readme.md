
1、下载原始视频
```
yt-dlp --cookies cookies.txt -o "15m.mp4"  https://www.youtube.com/watch?v=F2POpDZT8IY
```
2、生成节拍音频
```
python running_beat_generator.py --bpm 160 --duration 60
```
3、合并节拍音乐,节拍音乐增大音量3.0
```
ffmpeg -i 1hrs.mp4 -i running_beat_170bpm_60min.wav -filter_complex "[0:a]volume=1.0[bg];[1:a]volume=3.0[beat];[bg][beat]amix=inputs=2:duration=longest:dropout_transition=2[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac output_with_beat.mp4
```
4、截取教练的视频画面中间1/3部分内容
```
ffmpeg -i 1m.mp4 -filter:v "crop=iw/3:ih:iw/3:0" -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac -ar 44100 -b:a 128k middle_third_vertical.mp4

ffmpeg -i middle_third_vertical.mp4 -vf "scale=iw/2:ih/2" -an -c:v libx264 -crf 23 demostrate.mp4
```

5、画面叠加
```
ffmpeg -hwaccel auto -i 30m.mp4 -i demostrate.mp4 -filter_complex "[0:v][1:v]overlay=x=10:y=H-h-10:enable='between(mod(t,90),0,20)'" -c:v h264_nvenc -preset slow -cq 18 -c:a copy output.mp4
```

```
ffmpeg -hwaccel auto -i 30m.mp4 -stream_loop -1 -i tutorial.mp4 -filter_complex "[0:v][1:v]overlay=x=10:y=H-h-10" -t 1200 -c:v h264_nvenc -preset slow -cq 18 -c:a copy output.mp4
```
