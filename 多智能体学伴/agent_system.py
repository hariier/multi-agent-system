import os
import json
from datetime import datetime
import networkx as nx
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
        "default_model": "qwen-max"  # 也可选 qwen-plus, qwen-turbo
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPUAI_API_KEY",
        "default_model": "glm-4"  # 也可选 glm-4
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
# 1. 统一 LLM 客户端封装 (Universal Adapter)
# ==========================================
class UniversalLLMClient:
    def __init__(
        self, 
        provider: str = "deepseek", 
        model: Optional[str] = None, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        provider = provider.lower()
        config = LLM_PROVIDERS.get(provider, {})
        
        # 1. 决定 API Key
        env_var_name = config.get("env_key", f"{provider.upper()}_API_KEY")
        self.api_key = api_key or os.getenv(env_var_name)
        if not self.api_key:
            raise ValueError(f"❌ 未找到 {provider} 的 API Key，请设置环境变量 '{env_var_name}' 或显式传入！")
        
        # 2. 决定 Base URL 与 Model
        self.base_url = base_url or config.get("base_url")
        if not self.base_url:
            raise ValueError(f"❌ 未指定 {provider} 的 base_url！")
            
        self.model = model or config.get("default_model", provider)
        self.provider_name = provider

        # 3. 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """通用接口调用方法"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1 if json_mode else 0.5
        }
        
        # 针对支持 JSON mode 的模型开启格式约束
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
    max_chapter: int = 99
    
    intent_type: str = ""
    keywords: List[str] = field(default_factory=list)
    
    pruned_subgraph_context: str = ""
    matched_nodes: List[str] = field(default_factory=list)
    
    heuristic_draft: str = ""
    
    eval_pass: bool = False
    eval_reasons: List[str] = field(default_factory=list)  # 记录历次不通过的理由
    retry_count: int = 0                                   # 记录重试迭代次数
    final_response: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 3. 知识检索与拓扑剪枝引擎
# ==========================================
class KnowledgeGraphEngine:
    def __init__(self, json_path_or_dict: Any):
        self.graph = nx.DiGraph()

        if isinstance(json_path_or_dict, str):
            with open(json_path_or_dict, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json_path_or_dict
        
        self._build_graph_from_tree(data)
    
    def _build_graph_from_tree(self, data: dict):
        """将课程 JSON 的树状结构自动映射转换为 NetworkX 有向图"""
        knowledge_tree = data.get("knowledge_tree", [])

        for ch in knowledge_tree:
            ch_id = ch["chapter_id"]
            ch_title = ch["chapter_title"]
            ch_node_id = f"ch_{ch_id}"

            # 1. 添加章节节点
            self.graph.add_node(
                ch_node_id,
                type="chapter",
                chapter=ch_id,
                name=ch_title,
                summary=ch_title
            )

            for t_idx, topic in enumerate(ch.get("topics", [])):
                tp_name = topic["topic"]
                tp_node_id = f"tp_{ch_id}_{t_idx}"

                # 2. 添加主题节点并建立章节->主题的关联
                self.graph.add_node(    
                    tp_node_id,
                    type="topic",
                    chapter=ch_id,
                    chapter_name=ch_title,
                    name=tp_name,
                    summary=f"属于 {ch_title} 的主题：{tp_name}"
                )
                self.graph.add_edge(ch_node_id, tp_node_id, relation="contains")

                # 3. 添加知识点节点，并建立依赖关系
                prev_kp_id = None
                for k_idx, kp_text in enumerate(topic.get("knowledge_points", [])):
                    kp_node_id = f"kp_{ch_id}_{t_idx}_{k_idx}"

                    self.graph.add_node(
                        kp_node_id,
                        type="knowledge_point",
                        chapter=ch_id,
                        chapter_name=ch_title,
                        topic_name=tp_name,
                        name=f"知识点: {kp_text[:15]}...",
                        summary=kp_text
                    )
                
                    self.graph.add_edge(tp_node_id, kp_node_id, relation="contains")
                    
                    if prev_kp_id:
                        self.graph.add_edge(prev_kp_id, kp_node_id, relation="prerequisite")
                    prev_kp_id = kp_node_id

    def find_target_nodes(self, keywords: List[str]) -> List[str]:
        """根据关键词检索知识节点"""
        if not keywords:
            return []
        
        matched = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "knowledge_point":
                continue
            
            summary = data.get("summary", "").lower()
            topic_name = data.get("topic_name", "").lower()

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in summary or kw_lower in topic_name:
                    matched.append(node_id)
        return list(set(matched))

    def prune_subgraph(self, target_node_ids: List[str], max_chapter: int = 99) -> str:
        """基于命中的知识点进行拓扑剪枝，提取必要的约束上下文"""
        selected_nodes = set()
        for target_id in target_node_ids:
            if target_id in self.graph:
                selected_nodes.add(target_id)
                selected_nodes.update(nx.ancestors(self.graph, target_id))

        valid_kps = []
        for nid in selected_nodes:
            node_data = self.graph.nodes[nid]
            if node_data.get('chapter', 0) <= max_chapter and node_data.get('type') == 'knowledge_point':
                valid_kps.append(node_data)

        if not valid_kps:
            return "【知识库约束】：未检索到关联知识点或相关知识点已超纲。"

        valid_kps.sort(key=lambda x: x['chapter'])

        context_str = "【知识库约束范围（回答必须严格限制在以下知识范围内，禁止使用范围外的知识）】：\n"
        for idx, node in enumerate(valid_kps, 1):
            context_str += f"{idx}. [第{node['chapter']}章 {node.get('chapter_name', '')} - {node.get('topic_name', '')}] {node['summary']}\n"
        return context_str


# ==========================================
# 4. 四个协同智能体定义
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


# --- Agent 2: 知识检索与剪枝智能体 ---
class KnowledgeRetrievalAgent:
    def __init__(self, kg_engine: KnowledgeGraphEngine):
        self.kg = kg_engine

    def run(self, state: AgentState) -> AgentState:
        targets = self.kg.find_target_nodes(state.keywords)
        state.matched_nodes = targets
        state.pruned_subgraph_context = self.kg.prune_subgraph(targets, max_chapter=state.max_chapter)
        return state


# --- Agent 3: 启发学习智能体 (支持反馈修正) ---
class HeuristicLearningAgent:
    SYSTEM_PROMPT = """你是一位专业的计算机网络课程助教智能体。你的任务是根据学生的问题类型，遵循对应的教学规约进行解答：

{context}

【当前问题类型】：{intent}

【教学规约与回答要求】：
1. **若当前问题类型为“理论概念”**：
   - 请直接给出完整、全面、结构清晰的最终解答。
   - 回答内容必须严格限制在上方【知识库约束范围】内，严禁包含或提及超出该范围的知识点。

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
        
        # 如果存在上一次 Agent 4 拒绝的反馈信息，追加给 Agent 3 进行自我修正
        if feedback:
            user_prompt += f"\n\n【⚠️ 上一次回答未通过教学督导审核】\n督导退回原因与修改指示：{feedback}\n请务必根据上述督导反馈意见，修改并重新生成符合规范的解答。"

        state.heuristic_draft = self.llm.chat(system_prompt, user_prompt)
        return state


# --- Agent 4: 教学评价智能体 (只做评判与反馈) ---
class PedagogicalEvaluationAgent:
    SYSTEM_PROMPT = """你是一位严格的教学督导，负责评估助教的回答是否符合教学规范。

【当前问题类型】：{intent}

【知识库约束范围】：
{context}

【助教的回复】：
{draft}

【评估与审查规则】：
1. **知识超纲审查 (has_out_of_scope)**：
   - 检查助教是否使用了【知识库约束范围】之外的未说明知识点。如果是，则 has_out_of_scope = true，pass = false。

2. **答案泄露审查 (has_leakage)**：
   - **若问题类型为“理论概念”**：助教直接给出完整标准答案属于合规行为，不算泄露（has_leakage = false）。
   - **若问题类型为“作业题目”**：助教是否直接输出了最终答案数值、最终结论、完整代码或选项？如果直接给出了最终答案，则属于答案泄露（has_leakage = true，pass = false）。

3. **评价反馈规则**：
   - 你**只需负责客观审核并给出判定**，不需要替助教重新撰写答案。
   - 如果 pass 为 false，请在 reason 中给出明确、具体的拒绝原因与修改建议（例如：“在作业题目中直接给出了最终数值结果，请擦除答案并改为提示性提问” 或 “使用了超纲的知识点：XXX”）。

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
# 5. 协同调度器 (Orchestrator - 反馈循环控制)
# ==========================================
class MultiAgentOrchestrator:
    def __init__(self, llm_client: UniversalLLMClient, json_path: str = None):
        self.llm_client = llm_client
        
        if json_path and os.path.exists(json_path):
            self.kg_engine = KnowledgeGraphEngine(json_path)
        else:
            print(f"❌ 错误：找不到路径 {json_path}")
            exit(1)
        
        self.agent1 = TaskDecompositionAgent(self.llm_client)
        self.agent2 = KnowledgeRetrievalAgent(self.kg_engine)
        self.agent3 = HeuristicLearningAgent(self.llm_client)
        self.agent4 = PedagogicalEvaluationAgent(self.llm_client)

    def run_pipeline(
        self, 
        student_query: str, 
        max_chapter: int = 99, 
        enable_pruning: bool = True, 
        enable_eval: bool = True,
        max_retries: int = 3  # 最大反馈修正轮数
    ) -> AgentState:
        state = AgentState(student_query=student_query, max_chapter=max_chapter)
        
        # 1. 任务拆解
        state = self.agent1.run(state)
        
        # 2. 知识检索与剪枝
        if enable_pruning:
            state = self.agent2.run(state)
        else:
            state.pruned_subgraph_context = "【无知识约束】模式（无剪枝）"
            
        # 3 & 4. Agent 3 与 Agent 4 的反馈生成循环 (Reflection Loop)
        feedback = None
        for retry in range(max_retries):
            state.retry_count = retry + 1
            
            # Agent 3 根据反馈（初次为 None）生成草稿
            state = self.agent3.run(state, feedback=feedback)
            
            # 如果关闭评估模式，直接以 Agent 3 的输出作为最终回答并结束
            if not enable_eval:
                state.eval_pass = True
                state.final_response = state.heuristic_draft
                break
            
            # Agent 4 进行合规评判
            state = self.agent4.run(state)
            
            # 如果通过，保留 Agent 3 的回答，跳出循环
            if state.eval_pass:
                state.final_response = state.heuristic_draft
                break
            else:
                # 未通过：获取 Agent 4 最新给出的不通过理由，准备传给下一次循环的 Agent 3
                feedback = state.eval_reasons[-1] if state.eval_reasons else "回答不符合教学规约"
        
        # 如果达到最大重试次数仍未通过，默认取 Agent 3 最后一次生成的回答
        if not state.eval_pass:
            state.final_response = state.heuristic_draft

        return state


# ==========================================
# 6. 批量处理数据集功能
# ==========================================
def load_dataset(file_path: str) -> pd.DataFrame:
    """兼容各种表头格式的数据集加载函数"""
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 清理 Column1 等临时表头行
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


def batch_process_dataset(
    orchestrator: MultiAgentOrchestrator,
    dataset_path: str = "数据集改版.csv",
    output_json_path: str = None,
    output_csv_path: str = None,
    max_samples: int = None,
    max_chapter: int = 99
):
    """
    批量处理数据集中的问题，自动保存与汇总指标
    """
    model_tag = orchestrator.llm_client.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 文件路径若未指定，自动补全 模型名 + 时间戳 后缀
    if not output_json_path:
        output_json_path = f"batch_results_{model_tag}_{timestamp}.json"
    if not output_csv_path:
        output_csv_path = f"batch_results_{model_tag}_{timestamp}.csv"

    print(f"📂 加载数据集: {dataset_path} | 当前运行模型: [{model_tag}]")
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)
    print(f"🚀 开始批量处理，共 {total_count} 条问题...\n")

    results = []
    
    for idx in range(total_count):
        row = df.iloc[idx]
        query = str(row.get('问题', '')).strip()
        gt_answer = str(row.get('解答', '')).strip()
        gt_category = str(row.get('问题分类', '')).strip()
        gt_calc_ans = str(row.get('答案（只用于计算题）', '')).strip()

        print(f"[{idx+1}/{total_count}] 处理问题: {query[:30]}...")

        try:
            # 运行多智能体 Pipeline
            state = orchestrator.run_pipeline(student_query=query, max_chapter=max_chapter)
            
            res_item = {
                "id": idx + 1,
                "model_name": model_tag,
                "question": query,
                "gt_category": gt_category,
                "gt_answer": gt_answer,
                "gt_calc_answer": gt_calc_ans if gt_calc_ans != 'nan' else "",
                # Agent 1
                "agent1_intent": state.intent_type,
                "agent1_keywords": state.keywords,
                # Agent 2
                "agent2_matched_nodes": state.matched_nodes,
                "agent2_pruned_context": state.pruned_subgraph_context,
                # Agent 3 & 4
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
            print(f"❌ [错误] 处理第 {idx+1} 条问题时出错: {e}")
            res_item = {
                "id": idx + 1,
                "model_name": model_tag,
                "question": query,
                "gt_category": gt_category,
                "gt_answer": gt_answer,
                "status": "error",
                "error_msg": str(e)
            }

        results.append(res_item)
        
        # 实时写入 JSON 文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 导出 CSV 方便在 Excel 中对比
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    # 输出统计报告
    success_items = [r for r in results if r.get("status") == "success"]
    print("\n" + "="*50)
    print(f"🎉 [{model_tag}] 批量处理完成！")
    print(f"总计处理: {len(results)} 条")
    print(f"成功: {len(success_items)} 条，失败: {len(results) - len(success_items)} 条")
    
    if success_items:
        pass_count = sum(1 for r in success_items if r.get("agent4_eval_pass"))
        leakage_count = sum(1 for r in success_items if r.get("agent4_has_leakage"))
        scope_count = sum(1 for r in success_items if r.get("agent4_has_out_of_scope"))
        
        print(f"规约通过率 (Pass Rate): {pass_count/len(success_items):.2%} ({pass_count}/{len(success_items)})")
        print(f"答案泄露率 (Leakage Rate): {leakage_count/len(success_items):.2%} ({leakage_count}/{len(success_items)})")
        print(f"知识超纲率 (Out-of-Scope Rate): {scope_count/len(success_items):.2%} ({scope_count}/{len(success_items)})")
        
    print(f"\n结果已保存至:\n 📄 JSON: {output_json_path}\n 📊 CSV:  {output_csv_path}")
    print("="*50 + "\n")


# ==========================================
# 7. 一键横向多模型跑板函数
# ==========================================
def run_multi_model_benchmark(
    target_providers: List[str], 
    json_path: str, 
    dataset_path: str, 
    max_samples: int = 3
):
    """
    依次调用不同的模型运行完全相同的测试代码
    """
    for provider in target_providers:
        print(f"\n==========================================")
        print(f"🤖 正在启动模型提供商测试: [{provider.upper()}]")
        print(f"==========================================")
        
        try:
            # 1. 实例化通用大模型客户端
            client = UniversalLLMClient(provider=provider)
            
            # 2. 实例化 Agent 调度器
            orchestrator = MultiAgentOrchestrator(llm_client=client, json_path=json_path)
            
            # 3. 运行批量处理
            batch_process_dataset(
                orchestrator=orchestrator,
                dataset_path=dataset_path,
                max_samples=max_samples,
                max_chapter=99
            )
        except Exception as e:
            print(f"⚠️ 跳过模型 [{provider}]，原因: {e}")


# ==========================================
# 8. 主运行入口
# ==========================================
if __name__ == "__main__":
    json_path = "knowledge_data/计算机网络课程知识.json"
    dataset_path = "数据集改版.csv"

    # 1. 在这里定义你想要对比测试的模型列表 (如: "deepseek", "qwen", "glm", "kimi")
    # 提示：运行前需确保对应的环境变量已设置 (DASHSCOPE_API_KEY, ZHIPUAI_API_KEY, MOONSHOT_API_KEY, DEEPSEEK_API_KEY)
    models_to_test = ["kimi"]

    # 2. 启动多模型对比评测
    # 提示：测试阶段可将 max_samples 设为 3，跑全量数据集时设为 None
    run_multi_model_benchmark(
        target_providers=models_to_test,
        json_path=json_path,
        dataset_path=dataset_path,
        max_samples=3
    )
