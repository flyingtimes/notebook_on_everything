# AI多模型内容生成与评估系统

一个强大的AI内容生成和评估工具，支持多个模板和多个模型的组合使用，并通过deepeval框架进行智能评估，帮助您找到最佳的AI生成内容。

## ✨ 主要特性

- 🎯 **多模板支持**: 自动读取prompts文件夹下的所有.prompts模板文件
- 🤖 **多模型并行**: 支持同时使用多个AI模型生成内容
- 📊 **智能评估**: 集成deepeval框架，自动评估生成内容质量
- 🏆 **最佳选择**: 自动识别并推荐评分最高的生成结果
- 💾 **结果保存**: 自动保存生成内容和评估结果
- 🔄 **灵活模式**: 支持重新生成或直接评估现有文件

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenRouter API密钥

### 安装依赖

```bash
pip install openai deepeval jinja2 python-dotenv