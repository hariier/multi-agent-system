import os
import json
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, Any, List
from openai import OpenAI

# ==========================================
# 0. 阿里百炼 LLM 客户端封装
# ==========================================
class BailianLLMClient:
    def __init__(self, api_key: str = None, model: str = "qwen-max"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY，请在环境变量中设置或显式传入！")
        
        # 接入阿里百炼 OpenAI 兼容 Base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """调用百炼大模型生成回答"""
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1 if json_mode else 0.5  # 评估提取任务低温，生成任务中温
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# ==========================================
# 1. 全局状态定义
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
    eval_reasons: List[str] = field(default_factory=list)
    final_response: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 2. 知识检索与拓扑剪枝引擎
# ==========================================
class KnowledgeGraphEngine:
    def __init__(self, kg_data: dict):
        self.graph = nx.DiGraph()
        self._build_graph(kg_data)

    def _build_graph(self, data: dict):
        for node in data.get('nodes', []):
            self.graph.add_node(node['id'], **node)
        for edge in data.get('edges', []):
            self.graph.add_edge(edge['source'], edge['target'], relation=edge.get('relation', 'prerequisite'))

    def find_target_nodes(self, keywords: List[str]) -> List[str]:
        matched = []
        for node_id, data in self.graph.nodes(data=True):
            for kw in keywords:
                if kw.lower() in data['name'].lower() or kw.lower() in data.get('summary', '').lower():
                    matched.append(node_id)
        return list(set(matched))

    def prune_subgraph(self, target_node_ids: List[str], max_chapter: int = 99) -> str:
        selected_nodes = set()
        for target_id in target_node_ids:
            if target_id in self.graph:
                selected_nodes.add(target_id)
                # 获取所有前置依赖节点 (Ancestors)
                selected_nodes.update(nx.ancestors(self.graph, target_id))

        valid_nodes = []
        for nid in selected_nodes:
            node_data = self.graph.nodes[nid]
            if node_data.get('chapter', 0) <= max_chapter:
                valid_nodes.append(node_data)

        if not valid_nodes:
            return "【知识库约束】：未检索到关联知识点。"

        context_str = "【知识库约束范围（回答必须严格限制在以下知识范围内，禁止使用范围外的知识）】：\n"
        for idx, node in enumerate(valid_nodes, 1):
            context_str += f"{idx}. [第{node['chapter']}章 {node.get('chapter_name', '')}] {node['name']}: {node['summary']}\n"
        return context_str


# ==========================================
# 3. 四个协同智能体定义
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

    def __init__(self, llm_client: BailianLLMClient):
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


# --- Agent 3: 启发学习智能体 ---
class HeuristicLearningAgent:
    SYSTEM_PROMPT = """你是一位计算机网络课程的助教智能体。你的使命是引导学生自主思考，而不是替学生做题。

{context}

【教学规约】：
1. 当前提问类型：{intent}。
2. 严格禁止直接给出代码、直接给出题目选项或完整的标准答案。
3. 只能使用上方【知识库约束范围】内的概念，严禁提及超出该范围的后续高级知识点。
4. 请使用苏格拉底式提问，指出关键突破口或引导学生分步思考。"""

    def __init__(self, llm_client: BailianLLMClient):
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        prompt = self.SYSTEM_PROMPT.format(
            context=state.pruned_subgraph_context,
            intent="理论概念" if state.intent_type == "theoretical_concept" else "作业题目"
        )
        state.heuristic_draft = self.llm.chat(prompt, state.student_query)
        return state


# --- Agent 4: 教学评价智能体 ---
class PedagogicalEvaluationAgent:
    SYSTEM_PROMPT = """你是一位严格的教学督导，评估助教的回答是否符合教学规范。

【知识库约束范围】：
{context}

【助教的回复】：
{draft}

【评估维度】：
1. 答案泄露 (has_leakage)：助教是否直接给出了作业题目的答案、完整代码或直接选项？
2. 知识超纲 (has_out_of_scope)：助教是否使用了【知识库约束范围】以外的未说明知识点？

请评估并输出以下 JSON 格式：
{{
    "pass": true/false,
    "has_leakage": true/false,
    "has_out_of_scope": true/false,
    "fine_tuned_response": "如果 pass 为 false，请修改助教回复使其合规；如果 pass 为 true，直接复制原助教回复"
}}"""

    def __init__(self, llm_client: BailianLLMClient):
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        prompt = self.SYSTEM_PROMPT.format(
            context=state.pruned_subgraph_context,
            draft=state.heuristic_draft
        )
        res = self.llm.chat(prompt, "请评估助教的回答", json_mode=True)
        try:
            data = json.loads(res)
            state.eval_pass = data.get("pass", False)
            state.final_response = data.get("fine_tuned_response", state.heuristic_draft)
            state.metrics["has_leakage"] = data.get("has_leakage", False)
            state.metrics["has_out_of_scope"] = data.get("has_out_of_scope", False)
        except Exception:
            state.eval_pass = True
            state.final_response = state.heuristic_draft
        return state


# ==========================================
# 4. 协同调度器 (Orchestrator)
# ==========================================
class MultiAgentOrchestrator:
    def __init__(self, api_key: str = None, kg_data: dict = None):
        # 初始化百炼 LLM 客户端（默认使用 qwen-max）
        self.llm_client = BailianLLMClient(api_key=api_key, model="qwen-max")
        self.kg_engine = KnowledgeGraphEngine(kg_data or {"nodes": [], "edges": []})
        
        self.agent1 = TaskDecompositionAgent(self.llm_client)
        self.agent2 = KnowledgeRetrievalAgent(self.kg_engine)
        self.agent3 = HeuristicLearningAgent(self.llm_client)
        self.agent4 = PedagogicalEvaluationAgent(self.llm_client)

    def run_pipeline(self, student_query: str, max_chapter: int = 99, enable_pruning: bool = True, enable_eval: bool = True) -> AgentState:
        state = AgentState(student_query=student_query, max_chapter=max_chapter)
        
        # 1. 任务拆解
        state = self.agent1.run(state)
        
        # 2. 知识检索与剪枝
        if enable_pruning:
            state = self.agent2.run(state)
        else:
            state.pruned_subgraph_context = "【无知识约束】模式（无剪枝）"
            
        # 3. 启发式生成
        state = self.agent3.run(state)
        
        # 4. 教学评价与规约控制
        if enable_eval:
            state = self.agent4.run(state)
        else:
            state.final_response = state.heuristic_draft
            state.eval_pass = True
            
        return state


# ==========================================
# 5. 测试运行示例
# ==========================================
if __name__ == "__main__":
    # 模拟构建一个计算机网络 Mini 知识图谱
    mock_kg = {
        "nodes": [
            {
                "id": "node_tcp",
                "name": "TCP三次握手",
                "chapter": 3,
                "chapter_name": "传输层",
                "summary": "建立连接时进行SYN, SYN-ACK, ACK三次报文交互，用于同步序号。"
            },
            {
                "id": "node_seq",
                "name": "序列号与确认号",
                "chapter": 3,
                "chapter_name": "传输层",
                "summary": "TCP用Seq和Ack编号保证可靠传输和顺序交付。"
            }
        ],
        "edges": [
            {"source": "node_seq", "target": "node_tcp", "relation": "prerequisite"}
        ]
    }

    # 实例化调度系统
    orchestrator = MultiAgentOrchestrator(kg_data=mock_kg)
    
    # 模拟学生提问
    test_query = "请直接告诉我TCP三次握手第二次握手时，服务器发给客户端的SYN和ACK标志位分别是什么？直接给答案谢谢！"
    
    print(f"--- 学生输入: {test_query} ---\n")
    result_state = orchestrator.run_pipeline(student_query=test_query, max_chapter=3)
    
    print(f"【Agent 1 意图识别】: {result_state.intent_type}, 关键词: {result_state.keywords}")
    print(f"\n【Agent 2 剪枝约束】:\n{result_state.pruned_subgraph_context}")
    print(f"\n【Agent 3 初版回复】:\n{result_state.heuristic_draft}")
    print(f"\n【Agent 4 评价结果】: 合规={result_state.eval_pass}, 泄露={result_state.metrics.get('has_leakage')}, 超纲={result_state.metrics.get('has_out_of_scope')}")
    print(f"\n【最终系统输出】:\n{result_state.final_response}")