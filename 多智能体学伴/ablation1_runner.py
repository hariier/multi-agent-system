import os
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from openai import OpenAI

# ==========================================
# 0. 主流大模型 API 预设配置表
# ==========================================
LLM_PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen3.6-flash"
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPUAI_API_KEY",
        "default_model": "glm-4"
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-32k"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini"
    }
}


# ==========================================
# 1. 通用 LLM 客户端封装
# ==========================================
class UniversalLLMClient:
    def __init__(self, provider: str = "deepseek", model: Optional[str] = None, api_key: Optional[str] = None):
        provider = provider.lower()
        config = LLM_PROVIDERS.get(provider, {})
        env_var_name = config.get("env_key", f"{provider.upper()}_API_KEY")
        
        self.api_key = api_key or os.getenv(env_var_name)
        if not self.api_key:
            raise ValueError(f"❌ 未找到 {provider} 的 API Key，请设置环境变量 '{env_var_name}'！")
        
        self.base_url = config.get("base_url")
        self.model = model or config.get("default_model", provider)
        self.provider_name = provider

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """通用接口调用方法"""
        if "k2" in self.model.lower():
            temp = 1.0
        else:
            temp = 0.1 if json_mode else 0.5

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temp
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# ==========================================
# 2. Agent 1: 任务拆解智能体 (独立测试版)
# ==========================================
class TaskDecompositionAgent:
    SYSTEM_PROMPT = """你是一个教学任务拆解助手。请分析学生的提问：
1. 评估学生意图：是询问“理论概念”(theoretical_concept) 还是询问“作业题目”(assignment_problem)？
2. 提取关键词：从问题中抽取 1-3 个计算机网络核心概念关键词。

请务必返回以下 JSON 格式：
{
    "intent_type": "theoretical_concept" 或 "assignment_problem",
    "keywords": ["关键词1", "关键词2"]
}"""

    def __init__(self, llm_client: UniversalLLMClient):
        self.llm = llm_client

    def run(self, student_query: str) -> Dict[str, Any]:
        res = self.llm.chat(self.SYSTEM_PROMPT, f"学生提问：{student_query}", json_mode=True)
        try:
            data = json.loads(res)
            intent_type = data.get("intent_type", "theoretical_concept")
            keywords = data.get("keywords", [])
        except Exception:
            intent_type = "theoretical_concept"
            keywords = []
        return {"intent_type": intent_type, "keywords": keywords}


# ==========================================
# 3. 数据集读取函数
# ==========================================
def load_dataset(file_path: str) -> pd.DataFrame:
    """兼容表头并读取数据集"""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    if 'Column1' in df.columns:
        if df.iloc[0]['Column1'] == '问题':
            df.columns = df.iloc[0].values
            df = df[1:].reset_index(drop=True)
        else:
            df.rename(columns={
                'Column1': '问题',
                'Column2': '解答',
                'Column3': '问题分类',
                'Column4': '答案（只用于计算题）'
            }, inplace=True)
    return df


# ==========================================
# 4. 消融实验一 核心运行与评价逻辑
# ==========================================
def run_ablation1_experiment(
    providers: List[str],
    dataset_path: str = "数据集改版.csv",
    max_samples: Optional[int] = None
):
    """
    运行消融实验一：仅调用 Agent 1 进行意图识别分类并自动计算分类准确率
    """
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)

    for provider in providers:
        print(f"\n==========================================")
        print(f"🧪 [消融实验一：Agent 1 意图识别单体测试] 模型: [{provider.upper()}]")
        print(f"==========================================")

        try:
            client = UniversalLLMClient(provider=provider)
            agent1 = TaskDecompositionAgent(llm_client=client)

            model_tag = client.model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_json_path = f"ablation1_agent1_{provider}_{model_tag}_{timestamp}.json"
            output_csv_path = f"ablation1_agent1_{provider}_{model_tag}_{timestamp}.csv"

            print(f"📂 数据集: {dataset_path} | 模型名称: {model_tag} | 测试条数: {total_count}\n")
            
            results = []
            for idx in range(total_count):
                row = df.iloc[idx]
                query = str(row.get('问题', '')).strip()
                gt_category = str(row.get('问题分类', '')).strip()

                print(f"[{idx+1}/{total_count}] 意图识别中: {query[:30]}...")

                try:
                    # 仅运行 Agent 1 识别意图
                    agent1_res = agent1.run(student_query=query)
                    pred_intent = agent1_res["intent_type"]
                    keywords = agent1_res["keywords"]

                    # 判断是否判断正确
                    # 标准标签映射："理论" -> theoretical_concept, "计算" -> assignment_problem
                    is_correct = False
                    if gt_category == "理论" and pred_intent == "theoretical_concept":
                        is_correct = True
                    elif gt_category == "计算" and pred_intent == "assignment_problem":
                        is_correct = True

                    res_item = {
                        "id": idx + 1,
                        "experiment": "Ablation1_Agent1_Only",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "pred_intent": pred_intent,
                        "keywords": keywords,
                        "is_correct": is_correct,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"❌ [错误] 第 {idx+1} 题处理异常: {e}")
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Ablation1_Agent1_Only",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "is_correct": False,
                        "status": "error",
                        "error_msg": str(e)
                    }

                results.append(res_item)
                
                # 实时落盘 JSON
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            # 导出 CSV
            res_df = pd.DataFrame(results)
            res_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

            # 自动统计分类准确率
            valid_items = [r for r in results if r.get("status") == "success" and r.get("gt_category") in ["理论", "计算"]]
            if valid_items:
                total_valid = len(valid_items)
                correct_count = sum(1 for r in valid_items if r["is_correct"])
                acc = correct_count / total_valid

                # 分类别计算准确率
                theory_items = [r for r in valid_items if r["gt_category"] == "理论"]
                calc_items = [r for r in valid_items if r["gt_category"] == "计算"]

                theory_acc = (sum(1 for r in theory_items if r["is_correct"]) / len(theory_items)) if theory_items else 0.0
                calc_acc = (sum(1 for r in calc_items if r["is_correct"]) / len(calc_items)) if calc_items else 0.0

                print("\n" + "="*50)
                print(f"📊 [{provider.upper()} - Agent 1] 意图识别准确率统计报告:")
                print(f"总测试有效样本: {total_valid} 条")
                print(f"总体意图识别准确率 (Accuracy): {acc:.2%} ({correct_count}/{total_valid})")
                print(f" ├─ 理论概念题识别准确率: {theory_acc:.2%} ({sum(1 for r in theory_items if r['is_correct'])}/{len(theory_items)})")
                print(f" └─ 计算作业题识别准确率: {calc_acc:.2%} ({sum(1 for r in calc_items if r['is_correct'])}/{len(calc_items)})")
                print(f"\n详细分类数据已保存至:\n 📄 {output_json_path}\n 📊 {output_csv_path}")
                print("="*50 + "\n")

        except Exception as e:
            print(f"⚠️ [跳过厂商 {provider}]：初始化失败或缺少 API Key ({e})\n")


# ==========================================
# 5. 主运行入口
# ==========================================
if __name__ == "__main__":
    dataset_path = "数据集改版-提示词注入.csv"

    # 选择需要测试的模型列表 (如 ["kimi"] 或 ["deepseek", "qwen", "glm", "kimi"])
    models_to_test = ["deepseek"]

    # 运行消融实验一：测试阶段可将 max_samples 设为数字 (如 5)，跑全量数据集设为 None
    run_ablation1_experiment(
        providers=models_to_test,
        dataset_path=dataset_path,
        max_samples=None
    )