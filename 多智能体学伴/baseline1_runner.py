import os
import json
from datetime import datetime
import pandas as pd
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
        "default_model": "qwen3.7-plus"
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
# 1. Direct LLM 客户端 (无 Prompt, 无约束)
# ==========================================
class DirectLLMClient:
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

    def query_raw(self, user_question: str) -> str:
        """纯净问答调用：不加任何 System Prompt，不加任何知识图谱提示"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_question}  # 只有唯一的 user 消息
            ]
        }
        
        # 兼容部分思考/推理模型的温度限制
        if "k2" not in self.model.lower():
            kwargs["temperature"] = 0.7

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# ==========================================
# 2. 数据集读取函数
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
# 3. Baseline 1 核心运行逻辑
# ==========================================
def run_baseline1_experiment(
    providers: List[str],
    dataset_path: str = "数据集改版.csv",
    max_samples: Optional[int] = None
):
    """
    遍历调用指定的各大厂商 LLM 运行 Baseline 1 测试
    """
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)

    for provider in providers:
        print(f"\n==========================================")
        print(f"🧪 [Baseline1 零提示词基线实验] 模型: [{provider.upper()}]")
        print(f"==========================================")

        try:
            client = DirectLLMClient(provider=provider)
            model_tag = client.model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_json_path = f"baseline1_{provider}_{model_tag}_{timestamp}.json"
            output_csv_path = f"baseline1_{provider}_{model_tag}_{timestamp}.csv"

            print(f"📂 数据集: {dataset_path} | 模型名称: {model_tag} | 处理条数: {total_count}\n")
            
            results = []
            for idx in range(total_count):
                row = df.iloc[idx]
                query = str(row.get('问题', '')).strip()
                gt_answer = str(row.get('解答', '')).strip()
                gt_category = str(row.get('问题分类', '')).strip()
                gt_calc_ans = str(row.get('答案（只用于计算题）', '')).strip()

                print(f"[{idx+1}/{total_count}] 直连调用 {provider}: {query[:30]}...")

                try:
                    # 裸模型回答，无任何附加逻辑
                    raw_response = client.query_raw(user_question=query)
                    
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Baseline1",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "gt_answer": gt_answer,
                        "gt_calc_answer": gt_calc_ans if gt_calc_ans != 'nan' else "",
                        "raw_llm_response": raw_response,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"❌ [错误] 遇到异常: {e}")
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Baseline1",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "gt_answer": gt_answer,
                        "status": "error",
                        "error_msg": str(e)
                    }

                results.append(res_item)
                
                # 实时落盘写 JSON
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            # 导出 CSV
            res_df = pd.DataFrame(results)
            res_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

            print(f"✅ [{provider}] Baseline1 跑板完成！文件已存至:\n 📄 {output_json_path}\n 📊 {output_csv_path}\n")

        except Exception as e:
            print(f"⚠️ [跳过厂商 {provider}]：初始化失败或缺少 API Key ({e})\n")


# ==========================================
# 4. 主运行入口
# ==========================================
if __name__ == "__main__":
    dataset_path = "数据集改版.csv"

    # 💡 在这里指定你想要运行 Baseline1 的模型列表
    # 支持: "deepseek", "qwen", "glm", "kimi", "openai"
    # 如果想单个单独跑，修改为 ["deepseek"] 即可
    target_models = ["deepseek","kimi","glm","qwen"]

    # 运行 Baseline 1 测试
    # 测试阶段建议设置 max_samples=3，全量实验时设为 None（读取全量 150-200 题）
    run_baseline1_experiment(
        providers=target_models,
        dataset_path=dataset_path,
        max_samples=None
    )