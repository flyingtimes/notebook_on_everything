import os  # 导入操作系统相关功能
import platform  # 导入平台识别模块，用于检测操作系统类型
import subprocess  # 导入子进程模块，用于启动Chrome浏览器
from dotenv import load_dotenv  # 导入环境变量加载模块
from pathlib import Path  # 导入路径处理模块
import asyncio  # 导入异步IO模块
import json  # 导入JSON处理模块
import sys  # 导入系统模块，用于错误输出和退出程序
import argparse  # 导入命令行参数解析模块
import urllib.parse  # 导入URL解析模块
from patchright.async_api import async_playwright, Page, BrowserContext  # 导入Playwright相关类

# 加载环境变量，override=True表示环境变量会覆盖已存在的系统环境变量
load_dotenv(override=True)

# 定义敏感数据字典，用于替换文本中的敏感信息
SENSITIVE_DATA = {}

# 添加命令行参数解析
def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='下载YouTube视频')
    parser.add_argument('-u', '--url', type=str, help='要下载的YouTube视频URL')
    return parser.parse_args()


def get_chrome_binary_path():
    """
    获取 Chrome 浏览器二进制文件的路径
    如果 .env 文件存在，则从中读取 CHROME_BINARY_PATH
    否则根据操作系统返回默认路径
    
    Returns:
        str: Chrome浏览器可执行文件的路径
        
    Raises:
        Exception: 如果操作系统不受支持
    """
    # 检查 .env 文件是否存在并加载
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv()
        chrome_path = os.getenv('CHROME_BINARY_PATH')
        if chrome_path:
            return chrome_path
    
    # 根据操作系统确定默认路径
    system = platform.system()
    if system == 'Windows':
        # Windows 默认 Chrome 路径
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    elif system == 'Darwin':  # macOS
        # macOS 默认 Chrome 路径
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system == 'Linux':
        # Linux 默认 Chrome 路径
        return "/usr/bin/google-chrome"
    else:
        # 不支持的操作系统
        raise Exception(f"不支持的操作系统: {system}")

def launch_chrome_with_remote_debugging(port=9222, user_data="./user_data"):
    """
    启动Chrome浏览器并开启远程调试端口
    
    Args:
        port (int): 远程调试端口号，默认为9222
        user_data (str): 用户数据目录，默认为./user_data
    
    Returns:
        subprocess.Popen: 启动的Chrome进程对象
    """
    # 获取Chrome浏览器路径
    chrome_path = get_chrome_binary_path()
    print(f"使用Chrome路径: {chrome_path}")
    
    # 构建用户数据目录的完整路径
    user_data_path = Path(__file__).resolve().parent.joinpath(user_data)
    # 确保用户数据目录存在
    user_data_path.mkdir(exist_ok=True)
    
    # 构建Chrome启动命令
    cmd = [
        chrome_path,
        f"--remote-debugging-port={port}",  # 设置远程调试端口
        f"--user-data-dir={user_data_path}"  # 设置用户数据目录
    ]
    
    # 启动Chrome进程
    print(f"启动Chrome，远程调试端口: {port}")
    process = subprocess.Popen(cmd)
    
    # 等待用户登录网站
    prompt = input("请先登录网站，如果不需要登录可跳过;按任意键继续...")
    return process

# Add this async function to handle the Playwright operations
async def save_cookies_async():
    """
    Asynchronously connect to Chrome and save cookies
    """
    async with async_playwright() as p:
        browser = None
        try:
            # Connect to the local running Chrome browser
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")    
            default_context = browser.contexts[0]  # Get the default context
            cookies = await default_context.cookies()
            with open('cookies.json', 'w') as f:
                json.dump(cookies, f)
            print("Cookies have been saved to file...")
        except Exception as e:
            print(f"Error saving cookies: {e}")
            return 1
        finally:
            if browser:
                await browser.close()
    return 0

def download_youtube_video(url, cookies_file='cookies.json'):
    """
    使用yt-dlp下载YouTube视频
    
    Args:
        url (str): 要下载的YouTube视频URL
        cookies_file (str): 包含cookies的文件路径
    
    Returns:
        int: 下载成功返回0，失败返回非0值
    """
    try:
        # 检查cookies文件是否存在
        if not os.path.exists(cookies_file):
            print(f"错误: Cookies文件 '{cookies_file}' 不存在")
            return 1
            
        # 将JSON格式的cookies转换为Netscape格式
        cookies_txt = 'cookies.txt'
        convert_json_to_netscape_cookies(cookies_file, cookies_txt)
        
        print(f"开始下载视频: {url}")
        # 构建yt-dlp命令
        cmd = [
            'yt-dlp',
            '--cookies', cookies_txt,
            url
        ]
        
        # 执行yt-dlp命令
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("视频下载成功!")
        else:
            print(f"视频下载失败，返回代码: {result.returncode}")
            
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"下载过程中出错: {e}")
        return e.returncode
    except Exception as e:
        print(f"发生未知错误: {e}")
        return 1

def convert_json_to_netscape_cookies(json_file, netscape_file):
    """
    将JSON格式的cookies转换为Netscape格式
    
    Args:
        json_file (str): JSON格式cookies文件路径
        netscape_file (str): 输出的Netscape格式cookies文件路径
    """
    try:
        # 读取JSON格式的cookies
        with open(json_file, 'r') as f:
            cookies = json.load(f)
        
        # 创建Netscape格式的cookies文件
        with open(netscape_file, 'w') as f:
            # 写入Netscape cookies文件头
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# https://curl.haxx.se/docs/http-cookies.html\n")
            f.write("# This file was generated by yt-dlp script\n\n")
            
            # 写入每个cookie
            for cookie in cookies:
                domain = cookie.get('domain', '')
                # 处理domain，确保格式正确
                if domain.startswith('.'):
                    domain_flag = 'TRUE'
                else:
                    domain_flag = 'FALSE'
                
                path = cookie.get('path', '/')
                secure = 'TRUE' if cookie.get('secure', False) else 'FALSE'
                
                # 确保过期时间是有效的整数
                expires = cookie.get('expires', 0)
                if expires <= 0:
                    expires = 2147483647  # 设置一个远期的过期时间
                else:
                    expires = int(expires)
                
                name = cookie.get('name', '')
                value = cookie.get('value', '')
                
                # Netscape格式: domain flag path secure expiry name value
                f.write(f"{domain}\t{domain_flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
                
        print(f"Cookies已转换为Netscape格式并保存到 {netscape_file}")
    except Exception as e:
        print(f"转换cookies格式时出错: {e}")
        raise

def check_cookies_and_user_data():
    """
    检查user_data文件夹和cookies.txt文件是否存在
    
    Returns:
        bool: 如果两者都存在返回True，否则返回False
    """
    # 获取当前脚本所在目录
    script_dir = Path(__file__).resolve().parent
    
    # 检查user_data文件夹是否存在
    user_data_path = script_dir / "user_data"
    user_data_exists = user_data_path.exists() and user_data_path.is_dir()
    
    # 检查cookies.txt文件是否存在
    cookies_txt_path = script_dir / "cookies.txt"
    cookies_txt_exists = cookies_txt_path.exists() and cookies_txt_path.is_file()
    
    if user_data_exists and cookies_txt_exists:
        print("检测到已存在的user_data文件夹和cookies.txt文件，将直接使用它们")
        return True
    else:
        if not user_data_exists:
            print("未检测到user_data文件夹")
        if not cookies_txt_exists:
            print("未检测到cookies.txt文件")
        return False

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取视频URL
    video_url = args.url
    if not video_url:
        video_url = input("请输入要下载的YouTube视频URL: ")
        if not video_url:
            print("未提供视频URL，退出程序")
            sys.exit(1)
    
    # 检查是否已存在cookies和user_data
    if check_cookies_and_user_data():
        # 如果已存在，直接下载视频
        download_youtube_video(video_url, cookies_file='cookies.json')
    else:
        # 如果不存在，启动Chrome获取cookies
        chrome_process = None
        try:
            # 启动Chrome并开启远程调试
            chrome_process = launch_chrome_with_remote_debugging()
            
            # 运行异步函数保存cookies
            exit_code = asyncio.run(save_cookies_async())
            if exit_code != 0:
                print("获取cookies失败，退出程序")
                sys.exit(1)
                
            # 下载视频
            download_youtube_video(video_url)
            
            print("操作完成。")
        except KeyboardInterrupt:
            # 处理用户中断（Ctrl+C）
            print("正在关闭Chrome...")
            print("程序已终止")
            sys.exit(1)
        finally:
            # 确保终止Chrome进程
            if chrome_process:
                chrome_process.terminate()
