


          
# Chrome 爬虫数据笔记本

## 项目简介

这是一个基于 Chrome 浏览器的网页爬虫工具，可以通过远程调试接口控制 Chrome 浏览器进行自动化操作，主要用于数据抓取和下载任务。本项目特别针对图书资源网站进行了优化，可以自动搜索和下载电子书。

## 功能特点

- 自动检测并使用系统中的 Chrome 浏览器
- 支持通过 `.env` 文件自定义 Chrome 路径
- 使用 Chrome 远程调试协议进行浏览器控制
- 自动处理文件下载
- 支持命令行参数指定搜索内容
- 跨平台支持（Windows、macOS、Linux）

## 环境要求

- Python 3.11 或更高版本
- Chrome 浏览器
- 依赖库：
  - argparse
  - asyncio
  - python-dotenv
  - playwright
  - pathlib

## 安装步骤

1. 克隆仓库到本地
2. 创建并激活虚拟环境
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
4. 配置环境变量（可选）
   - 复制 `env.example` 为 `.env`
   - 在 `.env` 文件中设置 Chrome 路径和用户数据目录

## 使用方法

### 启动 Chrome 远程调试

```bash
python main.py
```

这将自动启动 Chrome 浏览器并开启远程调试端口（默认为9222）。

### 使用命令行参数搜索特定图书

```bash
python main.py -n "图书名称"
```

### 手动启动 Chrome 远程调试

如果需要手动启动 Chrome 远程调试，可以参考 `enable-chrome-remote-debugging.md` 文件中的命令：

```bash
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=C:\Users\13802\code\notebook_on_everything\chrome-crawl-data-notebook\user_data
```

## 项目结构

- `main.py` - 主程序文件
- `.env` - 环境变量配置文件（需从 env.example 创建）
- `env.example` - 环境变量示例文件
- `.gitignore` - Git 忽略文件配置
- `pyproject.toml` - 项目依赖配置
- `.python-version` - Python 版本配置
- `enable-chrome-remote-debugging.md` - Chrome 远程调试启动指南
- `downloads/` - 下载文件保存目录（自动创建）
- `user_data/` - Chrome 用户数据目录（自动创建）

## 注意事项

1. 首次运行时，Chrome 会要求登录相关网站
2. 下载的文件将保存在 `downloads` 目录中
3. 如果遇到权限问题，请确保有足够的权限运行 Chrome 和写入文件

## 自定义配置

可以通过 `.env` 文件自定义以下配置：

- `CHROME_BINARY_PATH` - Chrome 浏览器可执行文件路径
- `USER_DATA_PATH` - Chrome 用户数据目录路径

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 许可证

[MIT License](LICENSE)

        