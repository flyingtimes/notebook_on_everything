要解决各类网站、社交媒体登陆后爬取数据的问题，需要用到浏览器的状态信息和cookies（youtube-dl使用cookies）

使用extract_cookies，这时候会弹出一个浏览器窗口，在这个窗口中登录你的账号，登录成功后，关闭浏览器窗口，回到命令行窗口，你会发现命令行窗口已经输出了cookies信息。
浏览器状态存储在user_data里面，cookies信息存储在cookies.txt里面。

这个时候可以使用：
```
yt-dlp --cookies cookies.txt -x --audio-format mp3 https://www.youtube.com/watch?v=_OvXR0Lqhbk
```
来下载视频字幕了。

使用extract_and_download可以一次性下载视频