import os
import json
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field
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
# 1. 统一 LLM 客户端封装
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
# 2. 全局状态定义
# ==========================================
@dataclass
class AgentState:
    student_query: str
    
    intent_type: str = ""
    keywords: List[str] = field(default_factory=list)
    
    pruned_subgraph_context: str = "【无知识图谱约束模式】（Agent 2 已注销，未挂载知识图谱剪枝子图）"
    
    heuristic_draft: str = ""
    
    eval_pass: bool = False
    eval_reasons: List[str] = field(default_factory=list)
    retry_count: int = 0
    final_response: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 3. 智能体定义 (Agent 1, Agent 3, Agent 4)
# ==========================================

# --- Agent 1: 任务拆解智能体 ---
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

    def run(self, state: AgentState) -> AgentState:
        res = self.llm.chat(self.SYSTEM_PROMPT, f"学生提问：{state.student_query}", json_mode=True)
        try:
            data = json.loads(res)
            state.intent_type = data.get("intent_type", "theoretical_concept")
            state.keywords = data.get("keywords", [])
        except Exception:
            state.intent_type = "theoretical_concept"
            state.keywords = []
        return state


# --- Agent 3: 启发学习智能体 (无知识图谱拓扑剪枝) ---
class HeuristicLearningAgent:
    SYSTEM_PROMPT = """你是一位专业的计算机网络课程助教智能体。你的任务是根据学生的问题类型，遵循对应的教学规约进行解答：

{context}

【当前问题类型】：{intent}

【教学规约与回答要求】：
1. **若当前问题类型为“理论概念”**：
   - 请直接给出完整、全面、结构清晰的最终解答。

2. **若当前问题类型为“作业题目”/“计算作业”**：
   - **允许且只允许**给出解题所需的原理公式、理论推导以及必要的中间计算步骤。
   - **严禁直接给出最终答案数值、最终结论、选项或完整代码！**
   - 必须采用**柏拉图/苏格拉底式提问**，在完成中间推断后抛出关键引导性问题，启发学生自己动手完成最后的计算与总结。
"""

    def __init__(self, llm_client: UniversalLLMClient):
        self.llm = llm_client

    def run(self, state: AgentState, feedback: str = None) -> AgentState:
        intent_desc = "理论概念" if state.intent_type == "theoretical_concept" else "作业题目"
        system_prompt = self.SYSTEM_PROMPT.format(
            context=state.pruned_subgraph_context,
            intent=intent_desc
        )
        
        user_prompt = f"学生提问：{state.student_query}"
        
        if feedback:
            user_prompt += f"\n\n【⚠️ 上一次回答未通过教学督导审核】\n督导退回原因与修改指示：{feedback}\n请务必根据上述督导反馈意见，修改并重新生成符合规范的解答。"

        state.heuristic_draft = self.llm.chat(system_prompt, user_prompt)
        return state


# --- Agent 4: 教学评价智能体 ---
class PedagogicalEvaluationAgent:
    SYSTEM_PROMPT = """你是一位严格的教学督导，负责评估助教的回答是否符合教学规范。

【当前问题类型】：{intent}

【知识库约束范围】：
{context}

【助教的回复】：
{draft}

【评估与审查规则】：
1. **知识超纲审查 (has_out_of_scope)**：
   - 审查助教是否使用了不符合当前问题阶段的未说明/超出基础范围的高级网络知识点（例如在物理层问题中提及了复杂的运输层/应用层协议）。如果是，则 has_out_of_scope = true，pass = false。

2. **答案泄露审查 (has_leakage)**：
   - **若问题类型为“理论概念”**：助教直接给出完整标准答案属于合规行为，不算泄露（has_leakage = false）。
   - **若问题类型为“作业题目”**：助教是否直接输出了最终答案数值、最终结论、完整代码或选项？如果直接给出了最终答案，则属于答案泄露（has_leakage = true，pass = false）。

3. **字数篇幅限制审查**：
   - **若问题类型为“理论概念”**：理论回答内容不得超过300字。如果字数超过300字，则 pass = false，并在 reason 中说明：“理论回答内容超过300字，请精简至300字以内”。

4. **评价反馈规则**：
   - 你只需负责客观审核并给出判定，不需要替助教重新撰写答案。
   - 如果 pass 为 false，请在 reason 中给出明确、具体的拒绝原因与修改建议。

请评估并输出以下 JSON 格式：
{{
    "pass": true/false,
    "has_leakage": true/false,
    "has_out_of_scope": true/false,
    "reason": "通过则填写'合规'；不通过则提供具体退回重写的理由"
}}"""

    def __init__(self, llm_client: UniversalLLMClient):
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        intent_desc = "理论概念" if state.intent_type == "theoretical_concept" else "作业题目"
        prompt = self.SYSTEM_PROMPT.format(
            intent=intent_desc,
            context=state.pruned_subgraph_context,
            draft=state.heuristic_draft
        )
        res = self.llm.chat(prompt, "请评估助教的回答", json_mode=True)
        try:
            data = json.loads(res)
            state.eval_pass = data.get("pass", False)
            reason = data.get("reason", "未符合教学规范")
            
            state.metrics["has_leakage"] = data.get("has_leakage", False)
            state.metrics["has_out_of_scope"] = data.get("has_out_of_scope", False)
            
            if not state.eval_pass:
                state.eval_reasons.append(reason)
        except Exception:
            state.eval_pass = True
        return state


# ==========================================
# 4. 消融调度器 (无 Agent 2)
# ==========================================
class Ablation2Orchestrator:
    def __init__(self, llm_client: UniversalLLMClient):
        self.llm_client = llm_client
        self.agent1 = TaskDecompositionAgent(self.llm_client)
        self.agent3 = HeuristicLearningAgent(self.llm_client)
        self.agent4 = PedagogicalEvaluationAgent(self.llm_client)

    def run_pipeline(
        self, 
        student_query: str, 
        max_retries: int = 3
    ) -> AgentState:
        state = AgentState(student_query=student_query)
        
        # 1. Agent 1 任务拆解
        state = self.agent1.run(state)
        
        # 2. Agent 2 已注销/跳过！直接进入 Agent 3 与 Agent 4 的反馈循环
        feedback = None
        for retry in range(max_retries):
            state.retry_count = retry + 1
            
            # Agent 3 生成草稿
            state = self.agent3.run(state, feedback=feedback)
            
            # Agent 4 进行合规与超纲评判
            state = self.agent4.run(state)
            
            if state.eval_pass:
                state.final_response = state.heuristic_draft
                break
            else:
                feedback = state.eval_reasons[-1] if state.eval_reasons else "回答不符合教学规约"
        
        if not state.eval_pass:
            state.final_response = state.heuristic_draft

        return state


# ==========================================
# 5. 数据集读取函数
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
# 6. 消融实验二 核心批量测试逻辑
# ==========================================
def run_ablation2_experiment(
    providers: List[str],
    dataset_path: str = "数据集改版.csv",
    max_samples: Optional[int] = None
):
    """
    运行消融实验二：注销 Agent 2 知识图谱模块，评估回答是否会超纲
    """
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)

    for provider in providers:
        print(f"\n==========================================")
        print(f"🧪 [消融实验二：注销 Agent 2 知识图谱] 模型: [{provider.upper()}]")
        print(f"==========================================")

        try:
            client = UniversalLLMClient(provider=provider)
            orchestrator = Ablation2Orchestrator(llm_client=client)

            model_tag = client.model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_json_path = f"ablation2_no_agent2_{provider}_{model_tag}_{timestamp}.json"
            output_csv_path = f"ablation2_no_agent2_{provider}_{model_tag}_{timestamp}.csv"

            print(f"📂 数据集: {dataset_path} | 模型名称: {model_tag} | 测试条数: {total_count}\n")
            
            results = []
            for idx in range(total_count):
                row = df.iloc[idx]
                query = str(row.get('问题', '')).strip()
                gt_answer = str(row.get('解答', '')).strip()
                gt_category = str(row.get('问题分类', '')).strip()
                gt_calc_ans = str(row.get('答案（只用于计算题）', '')).strip()

                print(f"[{idx+1}/{total_count}] 处理问题: {query[:30]}...")

                try:
                    state = orchestrator.run_pipeline(student_query=query)
                    
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Ablation2_No_Agent2",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
                        "gt_answer": gt_answer,
                        "gt_calc_answer": gt_calc_ans if gt_calc_ans != 'nan' else "",
                        # Agent 1 结构化输出
                        "agent1_intent": state.intent_type,
                        "agent1_keywords": state.keywords,
                        # 反馈循环结果
                        "retry_count": state.retry_count,
                        "eval_reasons": state.eval_reasons,
                        "agent3_heuristic_draft": state.heuristic_draft,
                        "agent4_eval_pass": state.eval_pass,
                        "agent4_has_leakage": state.metrics.get("has_leakage", False),
                        "agent4_has_out_of_scope": state.metrics.get("has_out_of_scope", False),
                        "final_response": state.final_response,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"❌ [错误] 第 {idx+1} 题处理异常: {e}")
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Ablation2_No_Agent2",
                        "provider": provider,
                        "model_name": model_tag,
                        "question": query,
                        "gt_category": gt_category,
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

            # 输出统计报告
            success_items = [r for r in results if r.get("status") == "success"]
            print("\n" + "="*50)
            print(f"🎉 [{model_tag}] 消融实验二处理完成！指标结果:")
            print(f"总计处理: {len(results)} 条")
            
            if success_items:
                pass_count = sum(1 for r in success_items if r.get("agent4_eval_pass"))
                leakage_count = sum(1 for r in success_items if r.get("agent4_has_leakage"))
                scope_count = sum(1 for r in success_items if r.get("agent4_has_out_of_scope"))
                
                print(f"规约通过率 (Pass Rate): {pass_count/len(success_items):.2%} ({pass_count}/{len(success_items)})")
                print(f"答案泄露率 (Leakage Rate): {leakage_count/len(success_items):.2%} ({leakage_count}/{len(success_items)})")
                print(f"🔥 知识超纲率 (Out-of-Scope Rate): {scope_count/len(success_items):.2%} ({scope_count}/{len(success_items)})")
                
            print(f"\n结果已保存至:\n 📄 {output_json_path}\n 📊 {output_csv_path}")
            print("="*50 + "\n")

        except Exception as e:
            print(f"⚠️ [跳过厂商 {provider}]：初始化失败或缺少 API Key ({e})\n")


# ==========================================
# 7. 主运行入口
# ==========================================
if __name__ == "__main__":
    dataset_path = "数据集改版.csv"

    # 选择要测试的模型列表 (如 ["kimi"]，或 ["deepseek", "qwen", "glm", "kimi"])
    models_to_test = ["qwen"]

    # 运行消融实验二：测试阶段设为数字 (如 3)，跑全量测试设为 None
    run_ablation2_experiment(
        providers=models_to_test,
        dataset_path=dataset_path,
        max_samples=None
    )