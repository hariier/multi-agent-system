import os
import json
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from openai import OpenAI

# ==========================================
# 0. 大模型 API 厂商预设配置表
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
            raise ValueError(f"❌ 未找到 {provider} 的 API Key，请配置环境变量 '{env_var_name}'！")
        
        self.base_url = config.get("base_url")
        self.model = model or config.get("default_model", provider)
        self.provider = provider
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """通用接口调用方法"""
        # 特殊模型温度参数兼容（如 kimi-k2 系列）
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
# 2. Agent 1: 任务拆解智能体
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
# 3. Agent 3: 启发学习智能体 (无知识图谱剪枝)
# ==========================================
class HeuristicLearningAgent:
    SYSTEM_PROMPT = """你是一位专业的计算机网络课程助教智能体。你的任务是根据学生的问题类型，遵循对应的教学规约进行解答：
【当前问题类型】：{intent}

【教学规约与回答要求】：
1. **若当前问题类型为“理论概念”**：
   - 请直接给出完整、全面、结构清晰的最终解答。

2. **若当前问题类型为“作业题目”/“计算作业”**：
   - **允许且只允许**给出解题所需的原理公式、理论推导以及必要的中间计算步骤。
   - **严禁直接给出最终答案数值、最终结论、选项或完整代码！**
   - 必须采用**柏拉图/苏格拉底式提问**，在完成中间推断后抛出关键引导性问题，启发学生自己动手完成最后的计算与总结。"""

    def __init__(self, llm_client: UniversalLLMClient):
        self.llm = llm_client

    def run(self, student_query: str, intent_type: str) -> str:
        intent_desc = "理论概念" if intent_type == "theoretical_concept" else "作业题目"
        system_prompt = self.SYSTEM_PROMPT.format(intent=intent_desc)
        user_prompt = f"学生提问：{student_query}"
        
        return self.llm.chat(system_prompt, user_prompt)


# ==========================================
# 4. 数据集读取函数
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
# 5. Baseline 3 批量跑板核心逻辑
# ==========================================
def run_baseline3_experiment(
    providers: List[str],
    dataset_path: str = "数据集改版.csv",
    max_samples: Optional[int] = None
):
    """
    遍历调用各大厂商 LLM 运行 Baseline 3 (Agent1 + Agent3) 管道测试
    """
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)

    for provider in providers:
        print(f"\n==========================================")
        print(f"🧪 [Baseline3 Agent1+Agent3 双智能体基线实验] 模型: [{provider.upper()}]")
        print(f"==========================================")

        try:
            client = UniversalLLMClient(provider=provider)
            agent1 = TaskDecompositionAgent(llm_client=client)
            agent3 = HeuristicLearningAgent(llm_client=client)

            model_tag = client.model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_json_path = f"baseline3_{provider}_{model_tag}_{timestamp}.json"
            output_csv_path = f"baseline3_{provider}_{model_tag}_{timestamp}.csv"

            print(f"📂 数据集: {dataset_path} | 模型名称: {model_tag} | 处理条数: {total_count}\n")
            
            results = []
            for idx in range(total_count):
                row = df.iloc[idx]
                query = str(row.get('问题', '')).strip()
                gt_answer = str(row.get('解答', '')).strip()
                gt_category = str(row.get('问题分类', '')).strip()
                gt_calc_ans = str(row.get('答案（只用于计算题）', '')).strip()

                print(f"[{idx+1}/{total_count}] Baseline3 处理: {query[:30]}...")

                try:
                    # 1. 运行 Agent 1 拆解意图
                    agent1_res = agent1.run(student_query=query)
                    intent_type = agent1_res["intent_type"]
                    keywords = agent1_res["keywords"]

                    # 2. 运行 Agent 3 启发式生成
                    agent3_res = agent3.run(student_query=query, intent_type=intent_type)
                    
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Baseline3_Agent1_Agent3",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "gt_answer": gt_answer,
                        "gt_calc_answer": gt_calc_ans if gt_calc_ans != 'nan' else "",
                        # Agent 1 字段
                        "agent1_intent": intent_type,
                        "agent1_keywords": keywords,
                        # Agent 3 字段 (最终回答)
                        "agent3_response": agent3_res,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"❌ [错误] 处理第 {idx+1} 题时出现异常: {e}")
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Baseline3_Agent1_Agent3",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "gt_answer": gt_answer,
                        "status": "error",
                        "error_msg": str(e)
                    }

                results.append(res_item)
                
                # 实时落盘 JSON 文件
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            # 导出 CSV 格式
            res_df = pd.DataFrame(results)
            res_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

            print(f"✅ [{provider}] Baseline3 跑板完成！文件已存至:\n 📄 {output_json_path}\n 📊 {output_csv_path}\n")

        except Exception as e:
            print(f"⚠️ [跳过厂商 {provider}]：初始化失败或缺少 API Key ({e})\n")


# ==========================================
# 6. 主运行入口
# ==========================================
if __name__ == "__main__":
    dataset_path = "数据集改版.csv"

    # 💡 选择需要测试的模型列表
    # 支持: "deepseek", "qwen", "glm", "kimi", "openai"
    target_models = [ "deepseek"]

    # 运行 Baseline 3 测试
    # 测试阶段设置 max_samples=3，全量测试设为 None
    run_baseline3_experiment(
        providers=target_models,
        dataset_path=dataset_path,
        max_samples=None
    )