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

def get_all_prompt_templates(prompts_dir: str = "prompts") -> Dict[str, str]:
    """读取prompts文件夹下的所有模板文件"""
    templates = {}
    
    if not os.path.exists(prompts_dir):
        print(f"❌ prompts目录 {prompts_dir} 不存在")
        return templates
    
    # 查找所有.prompts文件
    prompt_files = glob.glob(os.path.join(prompts_dir, "*.prompts"))
    
    if not prompt_files:
        print(f"❌ 在 {prompts_dir} 目录中没有找到任何.prompts文件")
        return templates
    
    for file_path in prompt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # 获取文件名（不包含扩展名）作为模板名
            template_name = os.path.splitext(os.path.basename(file_path))[0]
            templates[template_name] = content
            print(f"✅ 成功读取模板: {template_name} (长度: {len(content)} 字符)")
            
        except Exception as e:
            print(f"❌ 读取模板文件 {file_path} 时出错: {e}")
    
    return templates

def read_prompt_template(file_path: str = "prompts/2.prompts", subject: str = None) -> str:
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

def sanitize_filename(name: str) -> str:
    """清理名称，使其适合作为文件名"""
    # 替换不适合文件名的字符
    return name.replace("/", "_").replace(":", "_").replace("?", "_").replace("*", "_").replace("<", "_").replace(">", "_").replace("|", "_").replace('"', "_")

def generate_content_with_templates_and_models(templates: Dict[str, str], model_list: List[str], api_key: str, subject: str = None):
    """使用多个模板和多个模型生成内容并保存到input文件夹"""
    # 确保input文件夹存在
    input_dir = "./input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"创建输出目录: {input_dir}")
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total_combinations = len(templates) * len(model_list)
    current_combination = 0
    
    print(f"\n开始生成内容: {len(templates)} 个模板 × {len(model_list)} 个模型 = {total_combinations} 个组合\n")
    
    for template_name, template_content in templates.items():
        print(f"\n{'='*60}")
        print(f"正在处理模板: {template_name}")
        print(f"{'='*60}")
        
        # 使用Jinja2模板引擎处理模板
        try:
            template = Template(template_content)
            if subject:
                processed_template = template.render(subject=subject)
            else:
                processed_template = template_content
        except Exception as e:
            print(f"❌ 处理模板 {template_name} 时出错: {e}")
            continue
        
        template_results = []
        
        for i, model_name in enumerate(model_list, 1):
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] 模板: {template_name} | 模型: {model_name}")
            
            try:
                # 创建模型实例
                model = create_openrouter_model(model_name, api_key)
                
                # 生成内容
                print("正在生成内容...")
                generated_content = model.generate(processed_template)
                # 去掉空行
                generated_content = "\n".join(line for line in generated_content.split("\n") if line.strip())
                
                # 创建文件名：模板名_模型名_时间戳.txt
                safe_template_name = sanitize_filename(template_name)
                safe_model_name = sanitize_filename(model_name)
                filename = f"{safe_template_name}_{safe_model_name}_{timestamp}.txt"
                file_path = os.path.join(input_dir, filename)
                
                # 保存内容到文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(generated_content)
                
                print(f"✅ 内容已保存到: {file_path}")
                
                # 记录结果
                result = {
                    "template_name": template_name,
                    "model": model_name,
                    "filename": filename,
                    "file_path": file_path,
                    "content_length": len(generated_content),
                    "status": "success",
                    "content": generated_content  # 添加内容用于后续评估
                }
                template_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ 模板 {template_name} + 模型 {model_name} 生成失败: {e}")
                result = {
                    "template_name": template_name,
                    "model": model_name,
                    "filename": None,
                    "file_path": None,
                    "content_length": 0,
                    "status": "failed",
                    "error": str(e),
                    "content": None
                }
                template_results.append(result)
                all_results.append(result)
        
        # 显示当前模板的结果摘要
        successful_count = sum(1 for r in template_results if r["status"] == "success")
        failed_count = len(template_results) - successful_count
        print(f"\n模板 '{template_name}' 结果摘要:")
        print(f"  成功生成: {successful_count}/{len(model_list)}")
        print(f"  生成失败: {failed_count}/{len(model_list)}")
    
    return all_results

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
    
    print("=== AI多模型内容生成与评估系统 ===")
    print("支持多模板处理\n")
    
    if args.regen:
        print("🔄 模式：重新生成内容并评估")
    else:
        print("📁 模式：直接评估现有文件")
    
    print(f"📝 主题：{args.subject}\n")
    
    # 定义用户需求（基于主题）
    user_requirement = f"{args.subject}"
    
    # 1. 检查API密钥
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 错误: 请在.env文件中设置OPENROUTER_API_KEY环境变量")
        return
    
    # 2. 获取模型列表和模板（仅在需要生成内容时）
    model_list = []
    generation_results = []
    
    if args.regen:
        model_list = get_model_list()
        print(f"从环境变量读取到 {len(model_list)} 个模型:")
        for i, model in enumerate(model_list, 1):
            print(f"  {i}. {model}")
        print()
        
        # 3. 读取所有提示词模板
        templates = get_all_prompt_templates("prompts")
        if not templates:
            print("❌ 无法读取任何提示词模板，程序退出")
            return
        
        print(f"\n成功读取 {len(templates)} 个模板:")
        for template_name in templates.keys():
            print(f"  - {template_name}")
        print()
        
        # 4. 生成内容
        print("开始使用多个模板和多个模型生成内容...")
        generation_results = generate_content_with_templates_and_models(templates, model_list, api_key, args.subject)
        
        # 5. 显示生成结果摘要
        print("\n" + "=" * 60)
        print("=== 总体生成结果摘要 ===")
        print("=" * 60)
        
        successful_count = sum(1 for r in generation_results if r["status"] == "success")
        failed_count = len(generation_results) - successful_count
        
        print(f"总组合数: {len(generation_results)}")
        print(f"成功生成: {successful_count}")
        print(f"生成失败: {failed_count}")
        
        # 按模板分组显示结果
        template_summary = {}
        for result in generation_results:
            template_name = result["template_name"]
            if template_name not in template_summary:
                template_summary[template_name] = {"success": 0, "failed": 0}
            
            if result["status"] == "success":
                template_summary[template_name]["success"] += 1
            else:
                template_summary[template_name]["failed"] += 1
        
        print("\n各模板生成情况:")
        for template_name, counts in template_summary.items():
            total = counts["success"] + counts["failed"]
            print(f"  {template_name}: {counts['success']}/{total} 成功")
        
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
    
    try:
        eval_model = create_openrouter_model(eval_model_name, api_key)
        print(f"✅ 成功配置评估模型: {eval_model_name}")
    except Exception as e:
        print(f"❌ 配置评估模型失败: {e}")
        return
    
    # 8. 生成或加载评估标准
    evaluation_steps_file = "output/evaluation_steps.json"
    eval_requirement = user_requirement  # 默认使用用户需求作为评估需求
    
    if os.path.exists(evaluation_steps_file) and not args.regen:
        print(f"\n发现已存在的评估标准文件: {evaluation_steps_file}")
        use_existing = input("是否使用现有的评估标准？(y/n): ").strip().lower()
        
        if use_existing == 'y':
            evaluation_steps = load_evaluation_steps(evaluation_steps_file)
            if evaluation_steps:
                print("已加载现有评估标准")
                # 使用默认的用户需求作为评估需求
                print(f"使用评估需求: {eval_requirement}")
            else:
                print("加载失败，将重新生成")
                eval_requirement = input("请输入评估需求: ")
                evaluation_steps = None
        else:
            eval_requirement = input("请输入新的评估需求: ")
            evaluation_steps = None
    else:
        eval_requirement = input("请输入评估需求: ")
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
        eval_results = evaluate_texts(text_files, eval_requirement, evaluation_steps, eval_model)
    except Exception as e:
        print(f"❌ 评估过程失败: {e}")
        return
    
    # 10. 按分数排序并显示结果
    eval_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 50)
    print("=== 评估结果 (按分数排序) ===")
    print("=" * 50)
    
    for i, (filename, score, reason) in enumerate(eval_results, 1):
        print(f"\n{i}. 文件: {filename}")
        print(f"   分数: {score:.3f}")
        print(f"   评价: {reason[:200]}{'...' if len(reason) > 200 else ''}")
    
    # 11. 突出显示最佳结果
    if eval_results:
        best_filename, best_score, best_reason = eval_results[0]
        print("\n" + "=" * 50)
        print("🏆 最佳结果")
        print("=" * 50)
        print(f"文件名: {best_filename}")
        print(f"分数: {best_score:.3f}")
        print(f"评价: {best_reason}")
        
        # 显示最佳内容的预览
        best_content = text_files[best_filename]
        print(f"\n内容预览 (前500字符):")
        print("-" * 30)
        print(best_content[:500])
        if len(best_content) > 500:
            print("...")
        print("-" * 30)
    
    # 12. 保存详细评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"output/evaluation_results_{timestamp}.json"
    
    # 确保output文件夹存在
    output_dir = os.path.dirname(results_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    detailed_results = {
        "evaluation_time": datetime.now().isoformat(),
        "user_requirement": eval_requirement,
        "eval_model": eval_model_name,
        "total_files": len(text_files),
        "evaluation_steps": evaluation_steps,
        "results": [
            {
                "rank": i + 1,
                "filename": filename,
                "score": score,
                "reason": reason,
                "content": text_files[filename]
            }
            for i, (filename, score, reason) in enumerate(eval_results)
        ]
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 详细评估结果已保存到: {results_file}")
    print("\n✅ 程序执行完成！")

if __name__ == "__main__":
    main()