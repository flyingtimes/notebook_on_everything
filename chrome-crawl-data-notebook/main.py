import os
import platform
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import json
import os
import sys
import argparse
from pathlib import Path
import urllib.parse
from patchright.async_api import async_playwright, Page, BrowserContext
from patchright.async_api import Page

# Load environment variables
load_dotenv(override=True)

SENSITIVE_DATA = {}

# 添加命令行参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description='下载Z-Library图书')
    parser.add_argument('-n', '--name', type=str, help='要查找下载的图书名称')
    return parser.parse_args()


# --- Helper Function for Replacing Sensitive Data ---
def replace_sensitive_data(text: str, sensitive_map: dict) -> str:
	"""Replaces sensitive data placeholders in text."""
	if not isinstance(text, str):
		return text
	for placeholder, value in sensitive_map.items():
		replacement_value = str(value) if value is not None else ''
		text = text.replace(f'<secret>{placeholder}</secret>', replacement_value)
	return text


# --- Helper Function for Robust Action Execution ---
class PlaywrightActionError(Exception):
	"""Custom exception for errors during Playwright script action execution."""

	pass


async def _try_locate_and_act(page: Page, selector: str, action_type: str, text: str | None = None, step_info: str = '') -> None:
	"""
	Attempts an action (click/fill) with XPath fallback by trimming prefixes.
	Raises PlaywrightActionError if the action fails after all fallbacks.
	"""
	print(f'Attempting {action_type} ({step_info}) using selector: {repr(selector)}')
	original_selector = selector
	MAX_FALLBACKS = 50  # Increased fallbacks
	# Increased timeouts for potentially slow pages
	INITIAL_TIMEOUT = 10000  # Milliseconds for the first attempt (10 seconds)
	FALLBACK_TIMEOUT = 1000  # Shorter timeout for fallback attempts (1 second)

	try:
		locator = page.locator(selector).first
		if action_type == 'click':
			await locator.click(timeout=INITIAL_TIMEOUT)
			print('clicked...')
		elif action_type == 'fill' and text is not None:
			await locator.fill(text, timeout=INITIAL_TIMEOUT)
		else:
			# This case should ideally not happen if called correctly
			raise PlaywrightActionError(f"Invalid action_type '{action_type}' or missing text for fill. ({step_info})")
		print(f"  Action '{action_type}' successful with original selector.")
		await page.wait_for_timeout(500)  # Wait after successful action
		return  # Successful exit
	except Exception as e:
		print(f"  Warning: Action '{action_type}' failed with original selector ({repr(selector)}): {e}. Starting fallback...")

		# Fallback only works for XPath selectors
		if not selector.startswith('xpath='):
			# Raise error immediately if not XPath, as fallback won't work
			raise PlaywrightActionError(
				f"Action '{action_type}' failed. Fallback not possible for non-XPath selector: {repr(selector)}. ({step_info})"
			)

		xpath_parts = selector.split('=', 1)
		if len(xpath_parts) < 2:
			raise PlaywrightActionError(
				f"Action '{action_type}' failed. Could not extract XPath string from selector: {repr(selector)}. ({step_info})"
			)
		xpath = xpath_parts[1]  # Correctly get the XPath string

		segments = [seg for seg in xpath.split('/') if seg]

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
						await locator.clear(timeout=FALLBACK_TIMEOUT)
						await page.wait_for_timeout(100)
					except Exception as clear_error:
						print(f'    Warning: Failed to clear field during fallback ({step_info}): {clear_error}')
					await locator.fill(text, timeout=FALLBACK_TIMEOUT)

				print(f"    Action '{action_type}' successful with fallback selector: {repr(fallback_xpath)}")
				await page.wait_for_timeout(500)
				return  # Successful exit after fallback
			except Exception as fallback_e:
				print(f'    Fallback attempt {i} failed: {fallback_e}')
				if i == MAX_FALLBACKS:
					# Raise exception after exhausting fallbacks
					raise PlaywrightActionError(
						f"Action '{action_type}' failed after {MAX_FALLBACKS} fallback attempts. Original selector: {repr(original_selector)}. ({step_info})"
					)

	# This part should not be reachable if logic is correct, but added as safeguard
	raise PlaywrightActionError(f"Action '{action_type}' failed unexpectedly for {repr(original_selector)}. ({step_info})")

# --- End Helper Functions ---
async def run_generated_script(book_name=None):
    global SENSITIVE_DATA
    # 创建下载完成事件
    download_completed = asyncio.Event()
    
    async with async_playwright() as p:
        browser = None
        context = None
        page = None
        exit_code = 0 # Default success exit code
        try:
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")

            
            
            # Set up download handler
            default_context = browser.contexts[0]
            page = default_context.pages[0]
            # Set up download handler to save files to Downloads folder
            download_path = Path(__file__).resolve().parent.joinpath("downloads")   
            # 使用异步函数处理下载
            async def handle_download(download):
                print(f"正在下载: {download.suggested_filename}")
                await download.save_as(os.path.join(download_path, download.suggested_filename))
                print(f"下载完成: {download.suggested_filename}")
                # 设置下载完成事件，通知主程序可以退出
                download_completed.set()
            
            page.on("download", handle_download)
            # Initial page handling
            if default_context.pages:
                page = default_context.pages[0]
                print('Using initial page provided by context.')
            else:
                page = await default_context.new_page()
                print('Created a new page as none existed.')

            # --- Step 1 ---
            # Action 1
            print(f"导航到: https://101ml.fi (Step 1, Action 1)")
            await page.goto("https://101ml.fi", timeout=5000)
            await page.wait_for_load_state('load', timeout=5000)
            
            # --- Step 2 ---
            # Action 2
            search_term = book_name if book_name else "\u5c04\u96d5\u82f1\u96c4\u4f20"
            print(f"搜索图书: {search_term}")
            await _try_locate_and_act(page, "xpath=//html/body/div[2]/div/div/div[1]/form/div[1]/div/div[1]/input", "fill", text=replace_sensitive_data(search_term, SENSITIVE_DATA), step_info="Step 2, 输入搜索关键词")
            # Action 3
            await _try_locate_and_act(page, "xpath=//html/body/div[2]/div/div/div[1]/form/div[1]/div/div[2]/div/button", "click", step_info="Step 2, 点击搜索")
            await page.wait_for_timeout(5000)
            selector = "xpath=//*[@id=\"searchResultBox\"]//div[contains(@class,\"book-item\")][1]//z-bookcard"
            locator = await page.locator(selector).first.get_attribute('href')
            print("找到图书链接: https://101ml.fi"+locator)
            await page.goto("https://101ml.fi"+locator, timeout=50000)
            await page.wait_for_load_state('load', timeout=50000)

            # Action 4
            await _try_locate_and_act(page, "xpath=//a[@class=\"btn btn-default addDownloadedBook\"]", "click", step_info="Step 3, 下载")
            print("等待下载完成...")
            
            # 等待下载完成事件或超时
            try:
                # 设置超时时间为60秒
                await asyncio.wait_for(download_completed.wait(), timeout=60)
                print("下载已完成，准备退出程序...")
            except asyncio.TimeoutError:
                print("下载超时，程序将退出...")
            
        except PlaywrightActionError as pae:
            print(f'\n--- Playwright 操作错误: {pae} ---', file=sys.stderr)
            exit_code = 1
        except Exception as e:
            print(f'\n--- 发生意外错误: {e} ---', file=sys.stderr)
            import traceback
            traceback.print_exc()
            exit_code = 1
        finally:
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
        raise Exception(f"不支持的操作系统: {system}")

def launch_chrome_with_remote_debugging(port=9222,user_data="./user_data"):
    """
    启动Chrome浏览器并开启远程调试端口
    
    Args:
        port: 远程调试端口号，默认为9222
    
    Returns:
        subprocess.Popen: 启动的Chrome进程对象
    """
    chrome_path = get_chrome_binary_path()
    print(f"使用Chrome路径: {chrome_path}")
    user_data = Path(__file__).resolve().parent.joinpath(user_data)
    # 构建Chrome启动命令
    cmd = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data}"
    ]
    
    # 启动Chrome进程
    print(f"启动Chrome，远程调试端口: {port}")
    process = subprocess.Popen(cmd)
    prompt = input("请先登录网站，如果不需要登录可跳过;按任意键继续...")
    return process

if __name__ == "__main__":
    args = parse_arguments()
    # 启动Chrome并开启远程调试
    chrome_process = launch_chrome_with_remote_debugging()
    
    print("Chrome已启动，按Ctrl+C终止程序...")
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(run_generated_script(args.name))
        # 脚本执行完成后关闭Chrome
        print("脚本执行完成，正在关闭Chrome...")
        chrome_process.terminate()
    except KeyboardInterrupt:
        print("正在关闭Chrome...")
        chrome_process.terminate()
        print("程序已终止")
