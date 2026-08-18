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
    max_chapter: int = 99
    
    intent_type: str = ""
    keywords: List[str] = field(default_factory=list)
    
    pruned_subgraph_context: str = ""
    matched_nodes: List[str] = field(default_factory=list)
    
    heuristic_draft: str = ""
    final_response: str = ""


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

                self.graph.add_node(    
                    tp_node_id,
                    type="topic",
                    chapter=ch_id,
                    chapter_name=ch_title,
                    name=tp_name,
                    summary=f"属于 {ch_title} 的主题：{tp_name}"
                )
                self.graph.add_edge(ch_node_id, tp_node_id, relation="contains")

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
# 4. 智能体定义 (Agent 1, Agent 2, Agent 3)
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


# --- Agent 3: 启发学习智能体 (无 Agent 4 反馈机制) ---
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

    def run(self, state: AgentState) -> AgentState:
        intent_desc = "理论概念" if state.intent_type == "theoretical_concept" else "作业题目"
        system_prompt = self.SYSTEM_PROMPT.format(
            context=state.pruned_subgraph_context,
            intent=intent_desc
        )
        
        user_prompt = f"学生提问：{state.student_query}"
        state.heuristic_draft = self.llm.chat(system_prompt, user_prompt)
        state.final_response = state.heuristic_draft
        return state


# ==========================================
# 5. 消融调度器 (Agent 1 + Agent 2 + Agent 3，无 Agent 4)
# ==========================================
class Ablation3Orchestrator:
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

    def run_pipeline(
        self, 
        student_query: str, 
        max_chapter: int = 99
    ) -> AgentState:
        state = AgentState(student_query=student_query, max_chapter=max_chapter)
        
        # 1. Agent 1 任务拆解
        state = self.agent1.run(state)
        
        # 2. Agent 2 知识图谱拓扑剪枝
        state = self.agent2.run(state)
        
        # 3. Agent 3 启发式解答生成 (无 Agent 4 审查，一次性生成直接输出)
        state = self.agent3.run(state)

        return state


# ==========================================
# 6. 数据集读取与批量测试
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


def run_ablation3_experiment(
    providers: List[str],
    json_path: str = "knowledge_data/计算机网络课程知识.json",
    dataset_path: str = "数据集改版.csv",
    max_samples: Optional[int] = None
):
    """
    运行消融实验三：注销 Agent 4 督导评价智能体，评估答案泄露情况
    """
    df = load_dataset(dataset_path)
    total_count = len(df) if max_samples is None else min(len(df), max_samples)

    for provider in providers:
        print(f"\n==========================================")
        print(f"🧪 [消融实验三：注销 Agent 4 教学评价智能体] 模型: [{provider.upper()}]")
        print(f"==========================================")

        try:
            client = UniversalLLMClient(provider=provider)
            orchestrator = Ablation3Orchestrator(llm_client=client, json_path=json_path)

            model_tag = client.model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_json_path = f"ablation3_no_agent4_{provider}_{model_tag}_{timestamp}.json"
            output_csv_path = f"ablation3_no_agent4_{provider}_{model_tag}_{timestamp}.csv"

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
                        "experiment": "Ablation3_No_Agent4",
                        "provider": provider,
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
                        # Agent 3 (无 Agent 4 介入下的原始输出)
                        "final_response": state.final_response,
                        "status": "success"
                    }
                except Exception as e:
                    print(f"❌ [错误] 第 {idx+1} 题处理异常: {e}")
                    res_item = {
                        "id": idx + 1,
                        "experiment": "Ablation3_No_Agent4",
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

            print(f"✅ [{provider.upper()}] 消融实验三测试完成！文件已存至:\n 📄 {output_json_path}\n 📊 {output_csv_path}\n")

        except Exception as e:
            print(f"⚠️ [跳过厂商 {provider}]：初始化失败或缺少 API Key ({e})\n")


# ==========================================
# 7. 主运行入口
# ==========================================
if __name__ == "__main__":
    json_path = "knowledge_data/计算机网络课程知识.json"
    dataset_path = "数据集改版-提示词注入.csv"

    # 选择需要跑测试的模型列表 (如 ["kimi"] 或 ["deepseek", "qwen", "glm", "kimi"])
    models_to_test = ["deepseek", "glm", "kimi"]

    # 运行消融实验三：测试阶段设为数字 (如 3)，跑全量数据集设为 None
    run_ablation3_experiment(
        providers=models_to_test,
        json_path=json_path,
        dataset_path=dataset_path,
        max_samples=None
    )