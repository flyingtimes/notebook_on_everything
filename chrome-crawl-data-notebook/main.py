import os  # 导入操作系统相关功能
import platform  # 导入平台识别模块，用于检测操作系统类型
import subprocess  # 导入子进程模块，用于启动Chrome浏览器
from dotenv import load_dotenv  # 导入环境变量加载模块
from pathlib import Path  # 导入路径处理模块
import asyncio  # 导入异步IO模块
import json  # 导入JSON处理模块
import sys  # 导入系统模块，用于错误输出和退出程序
import argparse  # 导入命令行参数解析模块
from pathlib import Path  # 重复导入，可以删除
import urllib.parse  # 导入URL解析模块
from patchright.async_api import async_playwright, Page, BrowserContext  # 导入Playwright相关类
from patchright.async_api import Page  # 重复导入，可以删除

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
    parser = argparse.ArgumentParser(description='下载Z-Library图书')
    parser.add_argument('-n', '--name', type=str, help='要查找下载的图书名称')
    return parser.parse_args()


# --- 敏感数据替换辅助函数 ---
def replace_sensitive_data(text: str, sensitive_map: dict) -> str:
    """
    替换文本中的敏感数据占位符
    
    Args:
        text (str): 包含敏感数据占位符的文本
        sensitive_map (dict): 敏感数据映射字典
        
    Returns:
        str: 替换后的文本
    """
    if not isinstance(text, str):
        return text
    for placeholder, value in sensitive_map.items():
        replacement_value = str(value) if value is not None else ''
        text = text.replace(f'<secret>{placeholder}</secret>', replacement_value)
    return text


# --- 健壮的操作执行辅助函数 ---
class PlaywrightActionError(Exception):
    """
    Playwright脚本操作执行过程中的自定义异常类
    """
    pass


async def _try_locate_and_act(page: Page, selector: str, action_type: str, text: str | None = None, step_info: str = '') -> None:
    """
    尝试定位元素并执行操作，支持XPath选择器的回退机制
    
    Args:
        page (Page): Playwright页面对象
        selector (str): 用于定位元素的选择器
        action_type (str): 操作类型，'click'或'fill'
        text (str, optional): 当action_type为'fill'时要填充的文本
        step_info (str, optional): 步骤信息，用于日志记录
        
    Raises:
        PlaywrightActionError: 如果所有尝试都失败则抛出此异常
    """
    print(f'Attempting {action_type} ({step_info}) using selector: {repr(selector)}')
    original_selector = selector
    MAX_FALLBACKS = 50  # 增加回退次数
    # 增加超时时间以适应潜在的慢速页面
    INITIAL_TIMEOUT = 10000  # 第一次尝试的超时时间（10秒）
    FALLBACK_TIMEOUT = 1000  # 回退尝试的超时时间（1秒）

    try:
        # 尝试使用原始选择器定位元素
        locator = page.locator(selector).first
        if action_type == 'click':
            # 执行点击操作
            await locator.click(timeout=INITIAL_TIMEOUT)
            print('clicked...')
        elif action_type == 'fill' and text is not None:
            # 执行填充文本操作
            await locator.fill(text, timeout=INITIAL_TIMEOUT)
        else:
            # 如果操作类型无效或填充操作缺少文本，则抛出异常
            raise PlaywrightActionError(f"Invalid action_type '{action_type}' or missing text for fill. ({step_info})")
        print(f"  Action '{action_type}' successful with original selector.")
        await page.wait_for_timeout(500)  # 操作成功后等待500毫秒
        return  # 成功退出
    except Exception as e:
        # 原始选择器失败，记录警告并开始回退
        print(f"  Warning: Action '{action_type}' failed with original selector ({repr(selector)}): {e}. Starting fallback...")

        # 回退机制仅适用于XPath选择器
        if not selector.startswith('xpath='):
            # 如果不是XPath选择器，则立即抛出异常
            raise PlaywrightActionError(
                f"Action '{action_type}' failed. Fallback not possible for non-XPath selector: {repr(selector)}. ({step_info})"
            )

        # 解析XPath选择器
        xpath_parts = selector.split('=', 1)
        if len(xpath_parts) < 2:
            raise PlaywrightActionError(
                f"Action '{action_type}' failed. Could not extract XPath string from selector: {repr(selector)}. ({step_info})"
            )
        xpath = xpath_parts[1]  # 正确获取XPath字符串

        # 分割XPath路径
        segments = [seg for seg in xpath.split('/') if seg]

        # 尝试回退，通过逐步简化XPath路径
        for i in range(1, min(MAX_FALLBACKS + 1, len(segments))):
            trimmed_xpath_raw = '/'.join(segments[i:])
            fallback_xpath = f'xpath=//{trimmed_xpath_raw}'

            print(f'    Fallback attempt {i}/{MAX_FALLBACKS}: Trying selector: {repr(fallback_xpath)}')
            try:
                locator = page.locator(fallback_xpath).first
                if action_type == 'click':
                    await locator.click(timeout=FALLBACK_TIMEOUT)
                elif action_type == 'fill' and text is not None:
                    try:
                        # 尝试清除字段内容
                        await locator.clear(timeout=FALLBACK_TIMEOUT)
                        await page.wait_for_timeout(100)
                    except Exception as clear_error:
                        print(f'    Warning: Failed to clear field during fallback ({step_info}): {clear_error}')
                    # 填充文本
                    await locator.fill(text, timeout=FALLBACK_TIMEOUT)

                print(f"    Action '{action_type}' successful with fallback selector: {repr(fallback_xpath)}")
                await page.wait_for_timeout(500)
                return  # 回退成功后退出
            except Exception as fallback_e:
                print(f'    Fallback attempt {i} failed: {fallback_e}')
                if i == MAX_FALLBACKS:
                    # 耗尽所有回退尝试后抛出异常
                    raise PlaywrightActionError(
                        f"Action '{action_type}' failed after {MAX_FALLBACKS} fallback attempts. Original selector: {repr(original_selector)}. ({step_info})"
                    )

    # 如果逻辑正确，这部分代码不应该被执行到，但作为安全保障添加
    raise PlaywrightActionError(f"Action '{action_type}' failed unexpectedly for {repr(original_selector)}. ({step_info})")

# --- 辅助函数结束 ---

async def run_generated_script(book_name=None):
    """
    运行生成的脚本，搜索并下载指定图书
    
    Args:
        book_name (str, optional): 要搜索的图书名称，如果为None则使用默认值
        
    Returns:
        None
    """
    global SENSITIVE_DATA
    # 创建下载完成事件，用于通知主程序下载已完成
    download_completed = asyncio.Event()
    
    async with async_playwright() as p:
        browser = None
        context = None
        page = None
        exit_code = 0  # 默认成功退出代码
        try:
            # 连接到本地运行的Chrome浏览器
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
            
            # 设置下载处理器
            default_context = browser.contexts[0]  # 获取默认上下文
            page = default_context.pages[0]  # 获取默认页面
            # 设置下载文件保存到Downloads文件夹
            download_path = Path(__file__).resolve().parent.joinpath("downloads")
            # 确保下载目录存在
            download_path.mkdir(exist_ok=True)
            
            # 定义异步下载处理函数
            async def handle_download(download):
                """
                处理下载事件的异步函数
                
                Args:
                    download: Playwright下载对象
                    
                Returns:
                    None
                """
                try:
                    print(f"正在下载: {download.suggested_filename}")
                    # 保存文件到指定路径
                    await download.save_as(os.path.join(download_path, download.suggested_filename))
                    print(f"下载完成: {download.suggested_filename}")
                    # 设置下载完成事件，通知主程序可以退出
                    download_completed.set()
                except Exception as e:
                    print(f"下载失败: {e}")
                    download_completed.set()  # 即使失败也设置事件，避免程序卡住
            
            # 注册下载事件处理函数
            page.on("download", handle_download)
            
            # 初始页面处理
            if default_context.pages:
                page = default_context.pages[0]
                print('Using initial page provided by context.')
            else:
                page = await default_context.new_page()
                print('Created a new page as none existed.')

            # --- 步骤1: 导航到网站 ---
            print(f"导航到: https://101ml.fi (Step 1, Action 1)")
            # 导航到目标网站
            await page.goto("https://101ml.fi", timeout=30000)  # 设置30秒超时
            # 等待页面加载完成
            await page.wait_for_load_state('load', timeout=30000)  # 设置30秒超时
            
            # --- 步骤2: 搜索图书 ---
            # 设置搜索关键词，如果未提供则使用默认值
            search_term = book_name if book_name else "\u5c04\u96d5\u82f1\u96c4\u4f20"  # 默认搜索"射雕英雄传"
            print(f"搜索图书: {search_term}")
            # 在搜索框中填入搜索关键词
            await _try_locate_and_act(page, "xpath=//html/body/div[2]/div/div/div[1]/form/div[1]/div/div[1]/input", "fill", text=replace_sensitive_data(search_term, SENSITIVE_DATA), step_info="Step 2, 输入搜索关键词")
            # 点击搜索按钮
            await _try_locate_and_act(page, "xpath=//html/body/div[2]/div/div/div[1]/form/div[1]/div/div[2]/div/button", "click", step_info="Step 2, 点击搜索")
            # 等待搜索结果加载
            await page.wait_for_timeout(30000)  # 等待30秒
            # 定位第一个搜索结果
            selector = "xpath=//*[@id=\"searchResultBox\"]//div[contains(@class,\"book-item\")][1]//z-bookcard"
            
            try:
                # 获取图书链接
                locator = await page.locator(selector).first.get_attribute('href')
                print("找到图书链接: https://101ml.fi"+locator)
                # 导航到图书详情页
                await page.goto("https://101ml.fi"+locator, timeout=30000)
                # 等待页面加载完成
                await page.wait_for_load_state('load', timeout=30000)

                # --- 步骤3: 下载图书 ---
                # 点击下载按钮
                await _try_locate_and_act(page, "xpath=//a[@class=\"btn btn-default addDownloadedBook\"]", "click", step_info="Step 3, 下载")
                print("等待下载完成...")
            except Exception as e:
                # 处理未找到图书或无法访问图书页面的情况
                print(f"未找到图书或无法访问图书页面: {e}")
                exit_code = 1
            
            # 等待下载完成事件或超时
            try:
                # 设置超时时间为60秒
                await asyncio.wait_for(download_completed.wait(), timeout=60)
                print("下载已完成，准备退出程序...")
            except asyncio.TimeoutError:
                # 处理下载超时情况
                print("下载超时，程序将退出...")
            
        except PlaywrightActionError as pae:
            # 处理Playwright操作错误
            print(f'\n--- Playwright 操作错误: {pae} ---', file=sys.stderr)
            exit_code = 1
        except Exception as e:
            # 处理其他意外错误
            print(f'\n--- 发生意外错误: {e} ---', file=sys.stderr)
            import traceback
            traceback.print_exc()
            exit_code = 1
        finally:
            # 清理资源，关闭浏览器和上下文
            print('\n--- 脚本执行完成 ---')
            print('正在关闭浏览器/上下文...')
            if context:
                 try: await context.close()
                 except Exception as ctx_close_err: print(f'  警告: 无法关闭上下文: {ctx_close_err}', file=sys.stderr)
            if browser:
                 try: await browser.close()
                 except Exception as browser_close_err: print(f'  警告: 无法关闭浏览器: {browser_close_err}', file=sys.stderr)
            print('浏览器/上下文已关闭.')
            # 使用确定的退出代码退出
            if exit_code != 0:
                print(f'脚本执行出错 (退出代码 {exit_code}).', file=sys.stderr)
                sys.exit(exit_code)

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

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 启动Chrome并开启远程调试
    chrome_process = launch_chrome_with_remote_debugging()
    
    print("Chrome已启动，按Ctrl+C终止程序...")
    try:
        # 在Windows系统上设置事件循环策略
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 运行生成的脚本
        asyncio.run(run_generated_script(args.name))
        
        # 脚本执行完成后关闭Chrome
        print("脚本执行完成，正在关闭Chrome...")
        chrome_process.terminate()
    except KeyboardInterrupt:
        # 处理用户中断（Ctrl+C）
        print("正在关闭Chrome...")
        chrome_process.terminate()
        print("程序已终止")
