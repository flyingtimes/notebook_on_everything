import os  # 导入操作系统相关功能
import platform  # 导入平台识别模块，用于检测操作系统类型
import subprocess  # 导入子进程模块，用于启动Chrome浏览器
from dotenv import load_dotenv  # 导入环境变量加载模块
from pathlib import Path  # 导入路径处理模块
import asyncio  # 导入异步IO模块
import json  # 导入JSON处理模块
from patchright.async_api import async_playwright  # 导入Playwright相关类

# 加载环境变量，override=True表示环境变量会覆盖已存在的系统环境变量
load_dotenv(override=True)

def get_chrome_binary_path():
    """
    获取Chrome浏览器二进制文件的路径
    如果.env文件存在，则从中读取CHROME_BINARY_PATH
    否则根据操作系统返回默认路径
    
    Returns:
        str: Chrome浏览器可执行文件的路径
    """
    # 检查.env文件是否存在并加载
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv()
        chrome_path = os.getenv('CHROME_BINARY_PATH')
        if chrome_path:
            return chrome_path
    
    # 根据操作系统确定默认路径
    system = platform.system()
    if system == 'Windows':
        # Windows默认Chrome路径
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    elif system == 'Darwin':  # macOS
        # macOS默认Chrome路径
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system == 'Linux':
        # Linux默认Chrome路径
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
    prompt = input("请先登录YouTube网站，如果不需要登录可跳过；按任意键继续...")
    return process

async def save_cookies_async():
    """
    异步连接到Chrome并保存cookies到文件
    
    Returns:
        int: 成功返回0，失败返回1
    """
    async with async_playwright() as p:
        browser = None
        try:
            # 连接到本地运行的Chrome浏览器
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")    
            default_context = browser.contexts[0]  # 获取默认上下文
            cookies = await default_context.cookies()
            
            # 直接保存为Netscape格式的cookies文件
            convert_json_to_netscape_cookies(cookies, 'cookies.txt')
            print("Cookies已保存到cookies.txt文件")
        except Exception as e:
            print(f"保存cookies时出错: {e}")
            return 1
        finally:
            if browser:
                await browser.close()
    return 0

def convert_json_to_netscape_cookies(cookies, netscape_file):
    """
    将JSON格式的cookies转换为Netscape格式
    
    Args:
        cookies (list): JSON格式的cookies列表
        netscape_file (str): 输出的Netscape格式cookies文件路径
    """
    try:
        # 创建Netscape格式的cookies文件
        with open(netscape_file, 'w',encoding='utf-8') as f:
            # 写入Netscape cookies文件头

            
            # 写入每个cookie
            for cookie in cookies:
                domain = cookie.get('domain', '')
                # 检查domain是否以点开头
                initial_dot = domain.startswith('.')
                if initial_dot:
                    domain = domain[1:]  # 移除开头的点
                
                # 在Netscape格式中，domain_specified必须与initial_dot匹配
                domain_specified = 'TRUE' if initial_dot else 'FALSE'
                
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
                
                # Netscape格式: domain domain_specified path secure expiry name value
                f.write(f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
                
        print(f"Cookies已转换为Netscape格式并保存到 {netscape_file}")
    except Exception as e:
        print(f"转换cookies格式时出错: {e}")
        raise

if __name__ == "__main__":
    print("YouTube视频下载工具 - 仅保存cookies")
    print("="*50)
    print("本工具将启动Chrome浏览器，请在浏览器中登录YouTube")
    print("登录后，工具将自动保存cookies到cookies.txt文件")
    print("然后您可以使用yt-dlp或youtube-dl下载视频:")
    print("yt-dlp --cookies cookies.txt [视频URL]")
    print("="*50)
    
    # 启动Chrome并开启远程调试
    chrome_process = launch_chrome_with_remote_debugging()
    
    try:
        # 运行异步函数保存cookies
        exit_code = asyncio.run(save_cookies_async())
        
        if exit_code == 0:
            print("\n操作成功完成！")
            print("您现在可以使用以下命令下载YouTube视频:")
            print("yt-dlp --cookies cookies.txt [视频URL]")
        else:
            print("\n操作失败，请检查错误信息。")
    except KeyboardInterrupt:
        # 处理用户中断（Ctrl+C）
        print("\n正在关闭Chrome...")
        print("程序已终止")
    finally:
        # 确保终止Chrome进程
        if chrome_process:
            chrome_process.terminate()
