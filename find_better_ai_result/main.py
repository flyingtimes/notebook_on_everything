import os
import json
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
from dotenv import load_dotenv
load_dotenv()
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI
from jinja2 import Template

class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model: str, api_key: str):
        self.api_key = api_key
        super().__init__(model)
        # 确保使用完整的模型名称，覆盖父类可能的修改
        self.model_name = model
      
    def load_model(self):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
      
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
      
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
      
    def get_model_name(self):
        return self.model_name

def create_openrouter_model(model_name: str = "openai/gpt-4", api_key: str = None):
    """创建OpenRouter模型实例"""
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("请设置OPENROUTER_API_KEY环境变量或提供api_key参数")
      
    return OpenRouterModel(
        model=model_name,
        api_key=api_key
    )

def get_model_list() -> List[str]:
    """从环境变量获取模型列表"""
    model_list_str = os.getenv("MODEL_LIST", "openai/gpt-4")
    return [model.strip() for model in model_list_str.split(",") if model.strip()]

def read_prompt_template(file_path: str = "prompts/1.prompts", subject: str = None) -> str:
    """读取提示词模板文件并填充变量"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template_content = f.read().strip()
            
        # 使用Jinja2模板引擎处理模板
        template = Template(template_content)
        
        # 如果提供了subject参数，则进行变量替换
        if subject:
            content = template.render(subject=subject)
        else:
            content = template_content
            
        return content
    except FileNotFoundError:
        print(f"提示词文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取提示词文件时出错: {e}")
        return None

def sanitize_filename(model_name: str) -> str:
    """清理模型名称，使其适合作为文件名"""
    # 替换不适合文件名的字符
    return model_name.replace("/", "_").replace(":", "_").replace("?", "_").replace("*", "_")

def generate_content_with_models(prompt_template: str, model_list: List[str], api_key: str):
    """使用多个模型生成内容并保存到input文件夹"""
    # 确保input文件夹存在
    input_dir = "./input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"创建输出目录: {input_dir}")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, model_name in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] 正在使用模型: {model_name}")
        
        try:
            # 创建模型实例
            model = create_openrouter_model(model_name, api_key)
            
            # 生成内容
            print("正在生成内容...")
            generated_content = model.generate(prompt_template)
            # 去掉空行
            generated_content = "\n".join(line for line in generated_content.split("\n") if line.strip())
            safe_model_name = sanitize_filename(model_name)
            filename = f"{safe_model_name}_{timestamp}.txt"
            file_path = os.path.join(input_dir, filename)
            
            # 保存内容到文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(generated_content)
            
            print(f"✅ 内容已保存到: {file_path}")
            
            # 记录结果
            results.append({
                "model": model_name,
                "filename": filename,
                "file_path": file_path,
                "content_length": len(generated_content),
                "status": "success",
                "content": generated_content  # 添加内容用于后续评估
            })
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 生成失败: {e}")
            results.append({
                "model": model_name,
                "filename": None,
                "file_path": None,
                "content_length": 0,
                "status": "failed",
                "error": str(e),
                "content": None
            })
    
    return results

def read_text_files(directory: str = "./input") -> Dict[str, str]:
    """读取指定目录下的所有文本文件"""
    text_files = {}
      
    # 支持多种文本文件格式
    patterns = ["*.txt", "*.md", "*.text"]
      
    for pattern in patterns:
        file_pattern = os.path.join(directory, pattern)
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # 提取实际生成的内容（跳过文件头信息）
                    if "=" * 50 in content:
                        content = content.split("=" * 50, 1)[1].strip()
                    if content:  # 只保存非空文件
                        text_files[filename] = content
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
      
    return text_files

def generate_evaluation_standard(sample_text: str, user_requirement: str, model: OpenRouterModel) -> List[str]:
    """使用G-Eval生成评估标准"""
    print("正在生成评估标准...")
      
    # 创建G-Eval指标来生成评估步骤
    criteria = f"""
    根据用户需求评估AI生成文本的质量。用户需求：{user_requirement}
      
    """
      
    # 创建一个临时测试用例来生成评估步骤
    temp_test_case = LLMTestCase(
        input=user_requirement,
        actual_output=sample_text
    )
      
    temp_metric = GEval(
        name="文本质量评估",
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,  # 使用OpenRouter模型
        verbose_mode=False
    )
      
    # 运行一次以生成评估步骤
    temp_metric.measure(temp_test_case)
      
    return temp_metric.evaluation_steps

def evaluate_texts(text_files: Dict[str, str], user_requirement: str,   
                  evaluation_steps: List[str], model: OpenRouterModel) -> List[Tuple[str, float, str]]:
    """使用固定的评估步骤评估所有文本"""
    print("开始评估文本...")
      
    # 创建固定评估步骤的G-Eval指标
    quality_metric = GEval(
        name="文本质量评估",
        evaluation_steps=evaluation_steps,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,  # 使用OpenRouter模型
        threshold=0.7,
        verbose_mode=True 
    )
      
    results = []
      
    for filename, content in text_files.items():
        # 创建测试用例
        test_case = LLMTestCase(
            input=user_requirement,
            actual_output=content
        )
          
        # 评估
        score = quality_metric.measure(test_case)
        reason = quality_metric.reason
          
        results.append((filename, score, reason))
        print(f"评估文件: {filename}, 分数: {score:.3f}")
      
    return results

def save_evaluation_steps(steps: List[str], file_path: str = "output/evaluation_steps.json"):
    """保存评估步骤到JSON文件"""
    # 确保output文件夹存在
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    data = {
        "evaluation_steps": steps,
        "created_at": datetime.now().isoformat(),
        "version": "1.0"
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"评估步骤已保存到: {file_path}")

def load_evaluation_steps(file_path: str = "output/evaluation_steps.json") -> List[str]:
    """从JSON文件加载评估步骤"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["evaluation_steps"]
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
        return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AI多模型内容生成与评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                    # 直接评估input文件夹中的现有文件
  python main.py --regen            # 重新生成内容并评估
  python main.py --regen --subject "新的主题"  # 使用新主题重新生成内容
        """
    )
    
    parser.add_argument(
        "--regen", 
        action="store_true", 
        help="重新生成内容。如果不使用此参数，程序将直接评估input文件夹中的现有文件"
    )
    
    parser.add_argument(
        "--subject", 
        type=str, 
        default="如何提升二年级小学生的内驱力",
        help="生成内容的主题（默认：如何提升二年级小学生的内驱力）"
    )
    
    parser.add_argument(
        "--eval-model", 
        type=str, 
        help="指定评估模型（默认使用环境变量EVAL_MODEL或第一个生成模型）"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=== AI多模型内容生成与评估系统 ===\n")
    
    if args.regen:
        print("🔄 模式：重新生成内容并评估")
    else:
        print("📁 模式：直接评估现有文件")
    
    print(f"📝 主题：{args.subject}\n")
    
    # 定义用户需求（基于主题）
    user_requirement = f"{args.subject}"
    
    # 1. 检查API密钥（仅在需要生成内容时检查）
    api_key = None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 错误: 请在.env文件中设置OPENROUTER_API_KEY环境变量")
        return
    
    # 2. 获取模型列表（仅在需要生成内容时获取）
    model_list = []
    generation_results = []
    
    if args.regen:
        model_list = get_model_list()
        print(f"从环境变量读取到 {len(model_list)} 个模型:")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
        print()
        
        # 3. 读取提示词模板
        prompt_template = read_prompt_template("prompts/1.prompts", subject=args.subject)
        if not prompt_template:
            print("❌ 无法读取提示词模板，程序退出")
            return
        
        print(f"✅ 成功读取提示词模板 (长度: {len(prompt_template)} 字符)")
        print(f"提示词预览: {prompt_template[:100]}...\n")
        
        # 4. 生成内容
        print("开始使用多个模型生成内容...")
        generation_results = generate_content_with_models(prompt_template, model_list, api_key)
        
        # 5. 显示生成结果摘要
        print("\n=== 生成结果摘要 ===")
        successful_count = sum(1 for r in generation_results if r["status"] == "success")
        failed_count = len(generation_results) - successful_count
        
        print(f"总模型数: {len(generation_results)}")
        print(f"成功生成: {successful_count}")
        print(f"生成失败: {failed_count}")
        
        if successful_count == 0:
            print("❌ 没有成功生成任何内容，无法进行评估")
            return
    else:
        print("⏭️  跳过内容生成，直接使用现有文件")
    
    # 6. 开始评估阶段
    print("\n" + "=" * 50)
    print("开始评估阶段...")
    print("=" * 50)
    
    # 检查input文件夹是否存在文件
    if not os.path.exists("./input"):
        print("❌ input文件夹不存在")
        if not args.regen:
            print("💡 提示：使用 --regen 参数可以重新生成内容")
        return
    
    # 7. 读取生成的文本文件
    text_files = read_text_files("./input")
    if not text_files:
        print("❌ 在 ./input 目录中没有找到任何文本文件")
        if not args.regen:
            print("💡 提示：使用 --regen 参数可以重新生成内容")
        return
    
    print(f"\n找到 {len(text_files)} 个文本文件进行评估:")
    for filename in text_files.keys():
        print(f"  - {filename}")
    print()
    
    # 配置评估模型
    eval_model_name = args.eval_model or os.getenv("OPENROUTER_MODEL")
    if not eval_model_name:
        if model_list:
            eval_model_name = model_list[0]
        else:
            eval_model_name = "openai/gpt-4"  # 默认评估模型
    
    print(f"使用评估模型: {eval_model_name}")
    
    # 检查评估模型的API密钥
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ 错误: 评估阶段需要OPENROUTER_API_KEY环境变量")
            return
    
    try:
        eval_model = create_openrouter_model(eval_model_name, api_key)
        print(f"✅ 成功配置评估模型: {eval_model_name}")
    except Exception as e:
        print(f"❌ 配置评估模型失败: {e}")
        return
    
    # 8. 生成或加载评估标准
    evaluation_steps_file = "output/evaluation_steps.json"
    
    if os.path.exists(evaluation_steps_file) and not args.regen:
        print(f"\n发现已存在的评估标准文件: {evaluation_steps_file}")
        use_existing = input("是否使用现有的评估标准？(y/n): ").strip().lower()
        
        if use_existing == 'y':
            evaluation_steps = load_evaluation_steps(evaluation_steps_file)
            if evaluation_steps:
                print("已加载现有评估标准")
            else:
                print("加载失败，将重新生成")
                evaluation_steps = None
        else:
            eval_requirement = input("请输入新的评估需求: ")
            evaluation_steps = None
    else:
        eval_requirement = input("请输入新的评估需求: ")
        evaluation_steps = None
    
    # 如果没有评估标准，则生成新的
    if not evaluation_steps:
        # 使用第一个文本作为样本来生成评估标准
        sample_filename = list(text_files.keys())[0]
        sample_text = text_files[sample_filename]
        
        print(f"\n使用文件 '{sample_filename}' 作为样本生成评估标准...")
        try:
            print(f"使用 {eval_model.get_model_name()} 做评估标准的生成")
            evaluation_steps = generate_evaluation_standard(sample_text, eval_requirement, eval_model)
            # 保存评估标准
            save_evaluation_steps(evaluation_steps, evaluation_steps_file)
        except Exception as e:
            print(f"❌ 生成评估标准失败: {e}")
            return
    
    print(f"\n当前评估标准包含 {len(evaluation_steps)} 个步骤:")
    for i, step in enumerate(evaluation_steps, 1):
        print(f"  {i}. {step}")
    print()
    
    # 9. 评估所有文本
    try:
        eval_results = evaluate_texts(text_files, user_requirement, evaluation_steps, eval_model)
    except Exception as e:
        print(f"❌ 评估过程失败: {e}")
        return
    
    # 10. 排序并显示评估结果
    eval_results.sort(key=lambda x: x[1], reverse=True)  # 按分数降序排列
    
    print("\n=== 评估结果 ===")
    print(f"{'排名':<4} {'文件名':<30} {'分数':<8} {'评估理由'}")
    print("-" * 100)
    
    for rank, (filename, score, reason) in enumerate(eval_results, 1):
        print(f"{rank:<4} {filename:<30} {score:.3f}    {reason}")
    
    # 11. 显示最佳结果
    best_file, best_score, best_reason = eval_results[0]
    print(f"\n🏆 得分最高的文本:")
    print(f"文件名: {best_file}")
    print(f"分数: {best_score:.3f}")
    print(f"评估理由: {best_reason}")
    
    # 从文件名中提取模型名称（如果是生成的文件）
    best_model = "未知"
    if generation_results:
        for result in generation_results:
            if result["filename"] == best_file:
                best_model = result["model"]
                break
    else:
        # 尝试从文件名中提取模型信息
        if "_" in best_file:
            best_model = best_file.split("_")[0].replace("_", "/")
    
    print(f"生成模型: {best_model}")
    print(f"评估模型: {eval_model_name}")
    
    # 12. 保存详细结果
    # 确保output文件夹存在
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    results_file = f"output/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detailed_results = {
        "generation_time": datetime.now().isoformat(),
        "mode": "regenerate" if args.regen else "evaluate_existing",
        "subject": args.subject,
        "user_requirement": user_requirement,
        "prompt_file": "prompts/1.prompts" if args.regen else None,
        "generation_models": model_list if args.regen else [],
        "evaluation_model": eval_model_name,
        "evaluation_steps": evaluation_steps,
        "generation_results": generation_results if args.regen else [],
        "evaluation_results": [
            {
                "rank": rank,
                "filename": filename,
                "score": score,
                "reason": reason,
                "content": text_files[filename]
            }
            for rank, (filename, score, reason) in enumerate(eval_results, 1)
        ],
        "best_result": {
            "filename": best_file,
            "score": best_score,
            "reason": best_reason,
            "model": best_model
        }
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📋 详细结果已保存到: {results_file}")
    print(f"📁 生成的文件保存在: ./input/ 目录")
    print(f"📊 评估结果保存在: ./output/ 目录")
    
    # 13. 显示最佳内容预览
    print(f"\n📖 最佳内容预览 ({best_file}):")
    print("-" * 50)
    best_content = text_files[best_file]
    preview_length = 500
    if len(best_content) > preview_length:
        print(best_content[:preview_length] + "...")
    else:
        print(best_content)
    print("-" * 50)
    
    # 14. 显示使用提示
    if not args.regen:
        print("\n💡 提示：")
        print("  - 使用 --regen 参数可以重新生成内容")
        print("  - 使用 --subject \"新主题\" 可以指定不同的生成主题")
        print("  - 使用 --eval-model \"模型名\" 可以指定不同的评估模型")

if __name__ == "__main__":
    main()