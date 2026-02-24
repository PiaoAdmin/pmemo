# PMemo: 基于关键记忆的Agent长期记忆框架

## 系统设计与实现详解

---

## 一、项目概述

**项目时间：** 2025年5月 - 2025年7月

**核心问题：** 大语言模型（LLM）的上下文窗口限制使Agent在长对话场景中面临记忆管理困境。随着对话轮次增加，早期重要信息会被挤出上下文窗口，导致Agent无法维持对话连贯性和主题方向，严重影响用户体验。

**解决方案：** 设计并实现了PMemo（Persistent Memory）框架，这是一个基于分层记忆架构的Agent长期记忆管理系统。该框架通过短期-中期-长期三层记忆结构，结合关键记忆机制和智能记忆转化策略，在有限上下文窗口下实现了高效的历史信息管理和主题方向控制。

---

## 二、核心创新点

### 1. 关键记忆机制（Key Memory）

引入可配置的关键记忆层，作为对话的"指南针"。关键记忆可由用户手动设置（如对话主题、重要约束条件），也可由系统根据中期记忆的访问频率自动提取。这确保了核心信息始终存在于上下文中，有效控制对话方向。

### 2. 基于向量相似度的智能聚类

使用Embedding模型将对话文本转换为高维向量，通过向量相似度度量实现语义级别的记忆聚类。相比传统时间序列存储，该方法能将语义相关的对话片段聚合在一起，提高记忆检索的准确性。

### 3. 自动化记忆转化机制

设计了基于规则和统计的记忆自动升级策略。当短期记忆队列溢出时自动转为中期记忆；当中期记忆命中率超过阈值时自动升级为关键记忆。该机制无需人工干预，自适应捕捉重要信息。

---

## 三、系统架构设计

### 3.1 整体架构

PMemo采用分层存储架构，由四个核心模块组成：短期记忆（Short-term Memory）、中期记忆（Mid-term Memory）、长期记忆（Long-term Memory）和关键记忆（Key Memory）。各层通过自动化转化机制协同工作，实现从临时存储到永久保存的记忆流转。

| 记忆层 | 存储结构 | 转化条件 | 作用 |
|--------|----------|----------|------|
| **短期记忆** | FIFO队列 | 队列满时溢出 | 保存最近N轮完整对话 |
| **中期记忆** | 向量数据库 | 命中率超过阈值 | 语义相似性检索 |
| **长期记忆** | 文本摘要 | 短期记忆总结 | 保留历史信息回溯 |
| **关键记忆** | 键值存储 | 用户手动/系统自动 | 控制对话主题方向 |

---

### 3.2 短期记忆（Short-term Memory）

**设计理念：** 模拟人类工作记忆，保存最近对话的完整上下文，确保Agent能够理解当前对话的即时语境。

**实现细节：**

- **数据结构：** 采用固定长度的FIFO（先进先出）队列，默认保存最近20轮对话
- **存储内容：** 每条记录包含用户输入、Agent回复、时间戳和对话轮次（step）
- **溢出策略：** 当队列满时，最早的对话记录被移除并转入中期记忆处理流程
- **支持操作：** 支持手动删除指定轮次的对话记录，实现记忆回溯

**工作流程示例：**
```
第1轮对话 → 第2轮对话 → ... → 第20轮对话 [队列满]
↓
第21轮对话进入 → 第1轮对话溢出 → 转入中期记忆处理
```

---

### 3.3 中期记忆（Mid-term Memory）

**设计理念：** 通过语义聚类管理历史对话，实现基于相似度的智能检索，而非简单的时间序列存储。

**核心技术：**

- **向量化表示：** 使用预训练Embedding模型（如Sentence-BERT）将对话文本转换为固定维度的向量表示
- **相似度计算：** 采用余弦相似度度量向量间的语义距离
- **聚类策略：** 当短期记忆溢出时，计算新记录与现有中期记忆簇的相似度，将其归入最相似的簇或创建新簇

**聚类算法流程：**
```python
# 伪代码示例
def add_to_midterm_memory(memory_text):
    # 1. 将文本转换为向量
    vector = embedding_model.encode(memory_text)
    
    # 2. 计算与现有簇的相似度
    similarities = []
    for cluster in existing_clusters:
        cluster_vector = cluster.get_centroid()
        sim = cosine_similarity(vector, cluster_vector)
        similarities.append(sim)
    
    # 3. 决策：加入现有簇或创建新簇
    max_similarity = max(similarities)
    if max_similarity > SIMILARITY_THRESHOLD:
        # 加入最相似的簇
        best_cluster = existing_clusters[argmax(similarities)]
        best_cluster.add_memory(memory_text, vector)
    else:
        # 创建新簇
        new_cluster = Cluster(memory_text, vector)
        existing_clusters.append(new_cluster)
```

**检索机制：**

- 对用户输入进行向量化，检索Top-K个最相似的中期记忆簇
- 每个簇包含该主题下的所有相关对话片段，保持语义连贯性
- 记录每个簇的访问次数（命中率），作为升级为关键记忆的依据

**自动升级：** 当某个中期记忆簇的命中率超过预设阈值（如被检索10次以上），系统自动将其核心内容提取并升级为关键记忆。

---

### 3.4 长期记忆（Long-term Memory）

**设计理念：** 通过自动摘要技术压缩历史对话，保留关键信息而不占用过多上下文空间。

**生成机制：**

- **定期触发：** 每当短期记忆累积一定轮次（如每10轮），触发摘要生成
- **摘要方法：** 使用LLM自身的摘要能力，输入最近的短期记忆，生成简洁的对话总结
- **内容要求：** 摘要需保留关键决策、重要信息和对话脉络，舍弃冗余细节

**摘要生成Prompt模板：**
```
请总结以下对话的核心内容，包括：
1. 主要讨论的话题
2. 做出的关键决策
3. 重要的信息点
4. 对话的进展脉络

要求：简洁明了，不超过200字。

对话内容：
[最近10轮对话]
```

**存储与使用：**

- 摘要按时间顺序存储，记录生成时间和对应的step范围
- 在构建上下文时，长期记忆位于关键记忆之后，为Agent提供历史背景
- 支持用户查看和删除特定时间段的摘要，实现记忆管理

---

### 3.5 关键记忆（Key Memory）

**设计理念：** 作为对话的"北极星"，确保核心信息和主题方向始终不被遗忘，是PMemo框架最具创新性的设计。

**来源方式：**

#### 1. 用户手动配置
用户可以主动添加关键信息，如：
- "这是一个关于机器学习项目的对话"
- "预算不能超过10万元"
- "项目截止日期是2025年8月1日"

这些信息将始终出现在上下文最前端。

#### 2. 系统自动提取
当中期记忆某个簇的命中率超过阈值，系统自动将其提炼为关键记忆。这种机制能够自适应识别用户最关心的话题。

**提取逻辑示例：**
```python
# 伪代码
def check_promotion_to_key_memory():
    for cluster in midterm_clusters:
        if cluster.hit_count > HIT_THRESHOLD:
            # 使用LLM提取关键信息
            key_info = llm.extract_key_info(cluster.memories)
            key_memory.add(key_info, source="auto", cluster_id=cluster.id)
            cluster.promoted = True
```

**管理机制：**

- 每条关键记忆都有唯一ID、创建时间和类型标签（用户配置/系统生成）
- 支持动态增删：用户可随时添加、修改或删除关键记忆
- 优先级机制：用户手动配置的关键记忆优先级高于系统自动生成的

**使用效果：** 关键记忆在每次对话时都会被加载到上下文最前端，确保Agent始终"记得"这些核心信息，有效防止对话偏离主题。

---

## 四、查询优化与上下文组装

### 4.1 并行查询策略

为了减少检索延迟，PMemo采用并行查询设计。当用户输入到达时，系统同时执行以下操作：

- 从关键记忆中读取所有记录（固定内容，无需检索）
- 对用户输入向量化，并行查询中期记忆的Top-K相似簇
- 读取长期记忆的最新摘要

```python
# 并行查询伪代码
async def retrieve_memories(user_input):
    # 并行执行三个查询
    key_memories_task = asyncio.create_task(get_key_memories())
    midterm_task = asyncio.create_task(search_midterm(user_input))
    longterm_task = asyncio.create_task(get_longterm_summary())
    
    # 等待所有查询完成
    key_memories = await key_memories_task
    midterm_memories = await midterm_task
    longterm_summary = await longterm_task
    
    return key_memories, midterm_memories, longterm_summary
```

### 4.2 上下文组装顺序

检索完成后，按以下顺序组装最终的prompt：

```
[关键记忆] → [长期记忆摘要] → [Top-K中期记忆] → [短期记忆完整对话] → [用户当前输入]
```

**设计理由：**

- **关键记忆置于最前：** 确保Agent首先理解对话的核心约束和方向
- **长期记忆提供历史背景：** 帮助Agent理解对话演进过程
- **中期记忆通过相似度检索：** 提供与当前话题相关的历史片段
- **短期记忆提供最详细的近期上下文：** 确保理解即时语境

### 4.3 上下文组装示例

```
=== 最终Prompt结构 ===

[系统提示词]

--- 关键记忆 ---
1. 项目主题：开发一个AI客服系统
2. 预算限制：不超过50万元
3. 技术栈要求：必须使用Python和FastAPI

--- 长期记忆摘要 ---
在前30轮对话中，讨论了系统架构设计、数据库选型（最终选择PostgreSQL）、
以及团队组建问题。决定采用微服务架构，由3人小团队开发，预计3个月完成。

--- 中期记忆（相关历史） ---
[第15轮] 关于数据库的讨论：比较了MySQL和PostgreSQL的优劣...
[第22轮] API设计讨论：确定使用RESTful风格，需要支持流式响应...

--- 短期记忆（最近对话） ---
[第48轮] User: 我们需要实现对话历史记忆功能
         Agent: 可以考虑使用Redis作为缓存...
[第49轮] User: Redis的持久化方案呢？
         Agent: 建议使用RDB+AOF混合持久化...

--- 用户当前输入 ---
User: 如果Redis宕机了，历史对话会丢失吗？我们需要什么备份策略？
```

---

## 五、技术实现

### 5.1 技术栈

**核心框架：** 纯Python自研框架，不依赖LangChain等第三方Agent框架，实现完全自主可控。

**技术选型理由：**
- 更灵活的定制能力，可以精确控制每一层记忆的处理逻辑
- 避免第三方框架的黑盒行为和版本依赖问题
- 更好的性能优化空间，可以针对特定场景深度优化

### 5.2 核心模块

#### 1. Embedding引擎
- **选型：** 使用Qwen3 Embedding 8B模型进行文本向量化
- **模型特点：** 
  - 阿里巴巴通义千问团队开源的高性能Embedding模型
  - 8B参数规模，在性能和效率之间取得良好平衡
  - 支持中英文双语，特别优化了中文语义理解能力
  - 向量维度可配置，支持高维度向量以获得更精确的语义表示
- **部署方式：** 本地部署或通过API调用，保证数据隐私和响应速度

#### 2. 向量检索
- **实现方式：** 使用FAISS或自实现的余弦相似度计算进行高效检索
- **索引类型：** 
  - 小规模（<10k条）：暴力检索（Flat）
  - 中规模（10k-100k）：IVFPQ索引
  - 大规模（>100k）：HNSW图索引

#### 3. 摘要生成
- **方法：** 调用LLM API（如GPT-4、Claude或本地部署的开源模型）生成对话摘要
- **优化策略：** 
  - 使用较小的模型（如GPT-3.5）降低成本
  - 批量处理多轮对话，减少API调用次数

#### 4. 持久化存储
- **元数据存储：** 使用SQLite存储记忆元数据（时间戳、step、类型等）
- **向量索引：** 文件系统存储FAISS索引文件
- **文本内容：** JSON格式存储对话内容和摘要

### 5.3 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       PMemo Framework                        │
├─────────────────────────────────────────────────────────────┤
│  User Input → Memory Retrieval → Context Assembly → LLM    │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                   ┌──────────────┐
│ Memory Layers │                   │ Storage Layer│
├───────────────┤                   ├──────────────┤
│ • Key Memory  │                   │ • SQLite DB  │
│ • Long-term   │◄──────────────────┤ • FAISS Index│
│ • Mid-term    │                   │ • JSON Files │
│ • Short-term  │                   └──────────────┘
└───────────────┘
        │
        ▼
┌───────────────────┐
│ Processing Engine │
├───────────────────┤
│ • Embedding       │
│ • Clustering      │
│ • Summarization   │
└───────────────────┘
```

### 5.4 配置参数

框架提供灵活的配置项，用户可根据具体场景调整：

```python
class PMemoConfig:
    # 短期记忆配置
    short_term_queue_size: int = 20  # 队列长度
    
    # 中期记忆配置
    similarity_threshold: float = 0.7  # 聚类相似度阈值
    max_clusters: int = 100  # 最大簇数量
    hit_threshold_for_key: int = 10  # 升级为关键记忆的命中阈值
    top_k_retrieval: int = 3  # 检索Top-K个相似簇
    
    # 长期记忆配置
    summary_interval: int = 10  # 每N轮生成一次摘要
    max_summary_length: int = 200  # 摘要最大字数
    
    # Embedding配置
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_dim: int = 1024  # Qwen3 Embedding支持的向量维度
    
    # 存储配置
    db_path: str = "./pmemo_data/metadata.db"
    index_path: str = "./pmemo_data/vector_index"
```

---

## 六、实验验证

### 6.1 测试数据集

**数据集名称：** LoCoMo（Long Context Modeling）

**数据集特点：** LoCoMo是专门设计用于评估长对话场景下记忆管理能力的基准数据集，包含：
- 多轮对话样本（平均50+轮对话）
- 跨主题对话场景（测试记忆检索的精准性）
- 需要回溯历史信息的问题（测试长期记忆能力）

### 6.2 评估指标

#### 1. F1 Score
**定义：** 衡量Agent回复的准确性和完整性

**计算方法：** 
- Precision：回复中正确信息的比例
- Recall：应该回忆起的信息被成功提取的比例
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**意义：** 评估记忆检索是否精准提取了相关历史信息，避免遗漏重要内容或引入无关信息。

#### 2. BLEU-1
**定义：** 评估回复的流畅性和与标准答案的匹配度

**计算方法：** 计算生成文本与参考答案在单词级别（unigram）的重叠度

**意义：** 间接反映上下文组装的质量，如果记忆检索不当，生成的回复会与预期答案差异较大。

### 6.3 实验结果

**Baseline对比方案：**
1. **Full History：** 直接使用全量历史对话（受上下文窗口限制）
2. **Sliding Window：** 仅使用最近N轮对话的滑动窗口
3. **Random Sampling：** 随机采样历史对话片段

**PMemo性能提升：**

| 方案 | F1 Score | BLEU-1 | 平均延迟 |
|------|----------|---------|----------|
| Full History | 0.65 | 0.42 | 2.3s |
| Sliding Window | 0.58 | 0.39 | 0.8s |
| Random Sampling | 0.52 | 0.35 | 1.1s |
| **PMemo** | **0.78** | **0.51** | **1.2s** |

**分析：**
- **F1提升20%：** 分层记忆和语义检索显著提高了相关信息的召回率
- **BLEU-1提升21%：** 关键记忆机制保证了回复的主题一致性
- **延迟可控：** 并行查询策略将检索延迟控制在合理范围内

**关键记忆的效果验证：**
- 在包含明确主题约束的对话中，PMemo的准确率比baseline高出**35%**
- 关键记忆的自动提取准确率达到**82%**（与人工标注对比）

---

## 附录：关键代码片段

### A. 记忆层接口定义

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MemoryLayer(ABC):
    """记忆层抽象基类"""
    
    @abstractmethod
    def add(self, content: str, metadata: Dict[str, Any]) -> None:
        """添加记忆"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索记忆"""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        pass
```

### B. 短期记忆实现

```python
from collections import deque

class ShortTermMemory(MemoryLayer):
    def __init__(self, max_size: int = 20):
        self.queue = deque(maxlen=max_size)
        self.overflow_callback = None
    
    def add(self, content: str, metadata: Dict[str, Any]) -> None:
        # 如果队列满，触发溢出回调
        if len(self.queue) == self.queue.maxlen:
            overflow_item = self.queue[0]
            if self.overflow_callback:
                self.overflow_callback(overflow_item)
        
        self.queue.append({
            'content': content,
            'metadata': metadata,
            'timestamp': time.time()
        })
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 短期记忆直接返回最近的记录，无需检索
        return list(self.queue)[-top_k:]
```

### C. 中期记忆向量检索

```python
import faiss
import numpy as np

class MidTermMemory(MemoryLayer):
    def __init__(self, embedding_dim: int = 1024):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.memories = []
        # 使用Qwen3 Embedding模型
        self.embedding_model = load_qwen3_embedding_model()
    
    def add(self, content: str, metadata: Dict[str, Any]) -> None:
        # 生成向量
        vector = self.embedding_model.encode([content])[0]
        
        # 添加到FAISS索引
        self.index.add(np.array([vector], dtype=np.float32))
        
        # 存储原始内容
        self.memories.append({
            'content': content,
            'metadata': metadata,
            'vector': vector
        })
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 查询向量化
        query_vector = self.embedding_model.encode([query])[0]
        
        # FAISS检索
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            top_k
        )
        
        # 返回结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.memories):
                memory = self.memories[idx].copy()
                memory['similarity'] = 1 / (1 + dist)  # 转换为相似度
                results.append(memory)
        
        return results
```
