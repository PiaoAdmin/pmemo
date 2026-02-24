import json
import numpy as np
from typing import Optional, Dict, Any

try:
    from .utils import get_timestamp, get_embedding, normalize_vector, OpenAIClient, gpt_user_profile_analysis, gpt_knowledge_extraction
    from .storage_provider import ChromaStorageProvider
except ImportError:
    from utils import get_timestamp, get_embedding, normalize_vector, OpenAIClient, gpt_user_profile_analysis, gpt_knowledge_extraction
    from storage_provider import ChromaStorageProvider

class LongTermMemory:
    def __init__(self, 
                 storage_provider: ChromaStorageProvider, 
                 llm_interface: OpenAIClient,
                 knowledge_capacity=100, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 embedding_model_kwargs: Optional[dict] = None,
                 llm_model: str = "gpt-4o-mini",
                 use_embedding_api: bool = False):  # 新增：是否使用API调用embedding服务
        self.storage = storage_provider
        self.llm_interface = llm_interface
        self.knowledge_capacity = knowledge_capacity
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.llm_model = llm_model  # 保存模型名称
        self.use_embedding_api = use_embedding_api  # 保存 API 参数

    def update_user_profile(self, user_id: str, conversation_history: str) -> Optional[Dict[str, Any]]:
        """
        Generates a new user profile based on conversation history and updates it in storage.
        """
        existing_profile_str = json.dumps(self.get_user_profile(user_id) or {})
        
        updated_profile = gpt_user_profile_analysis(
            conversation_str=conversation_history,
            client=self.llm_interface,
            model=self.llm_model,  # 传递模型参数
            existing_user_profile=existing_profile_str
        )
        
        if updated_profile:
            self.storage.update_user_profile(user_id, updated_profile)
            print(f"LongTermMemory: Updated user profile for {user_id}.")
            return updated_profile
        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.get_user_profile(user_id)

    def add_knowledge(self, knowledge_text: str, knowledge_type: str = "user"):
        """
        Adds a knowledge entry (for user or assistant) to ChromaDB.
        knowledge_type can be 'user' or 'assistant'.
        """
        print(f"DEBUG: add_knowledge received text: '{knowledge_text}'") # Debugging line
        if not knowledge_text or knowledge_text.strip().lower() in ["", "none", "- none", "- none."]:
            print(f"LongTermMemory: Empty {knowledge_type} knowledge received, not saving.")
            return
        
        vec = get_embedding(
            knowledge_text, 
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        vec = normalize_vector(vec).tolist()
        
        if knowledge_type == "user":
            knowledge_id = self.storage.add_user_knowledge(knowledge_text, vec)
        else:
            knowledge_id = self.storage.add_assistant_knowledge(knowledge_text, vec)
        
        self.storage.enforce_knowledge_capacity(knowledge_type, self.knowledge_capacity)
        return knowledge_id

    def extract_knowledge_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to extract structured knowledge from a block of text.
        """
        if not text.strip():
            return None
        return gpt_knowledge_extraction(
            conversation_str=text, 
            client=self.llm_interface,
            model=self.llm_model  # 传递模型参数
        )

    def get_user_knowledge(self) -> list:
        return self.storage.get_all_user_knowledge()

    def get_assistant_knowledge(self) -> list:
        return self.storage.get_all_assistant_knowledge()

    def search_knowledge(self, query: str, knowledge_type: str = "user", top_k=5) -> list:
        query_vec = get_embedding(
            query, 
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec).tolist()
        
        if knowledge_type == "user":
            results = self.storage.search_user_knowledge(query_vec, top_k=top_k)
        else:
            results = self.storage.search_assistant_knowledge(query_vec, top_k=top_k)
        
        print(f"LongTermMemory: Searched {knowledge_type} knowledge for '{query[:30]}...'. Found {len(results)} matches.")
        return results

    # -------------------- Long-term summary blocks --------------------

    def add_summary_segment(
        self,
        text: str,
        step_start: int,
        step_end: int,
        source: str = "short_term_rollup",
        version: str = "v1"
    ) -> Optional[str]:
        if not text or not text.strip():
            return None

        vec = get_embedding(
            text,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        vec = normalize_vector(vec).tolist()

        ltm_id = f"ltm_seg_{step_start}_{step_end}"
        payload = {
            "ltm_id": ltm_id,
            "text": text.strip(),
            "step_start": step_start,
            "step_end": step_end,
            "timestamp": get_timestamp(),
            "source": source,
            "status": "active",
            "version": version
        }
        self.storage.add_long_term_summary(payload, vec)
        self.storage.record_memory_event(
            action="add",
            memory_type="long_term_summary",
            memory_id=ltm_id,
            detail={"step_start": step_start, "step_end": step_end, "source": source}
        )
        return ltm_id

    def search_long_term_summary(self, query: str, top_k: int = 5, threshold: float = 0.05) -> list:
        if not query or not query.strip():
            return []
        query_vec = get_embedding(
            query,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec).tolist()
        results = self.storage.search_long_term_summaries(query_vec, top_k=top_k, status="active")
        return [item for item in results if item.get("similarity", 0.0) >= threshold]

    def list_long_term_summaries(self, status: str = "active", limit: int = 20) -> list:
        return self.storage.list_long_term_summaries(status=status, limit=limit)

    # Key user operation: delete long-term summary blocks manually.
    def delete_summary_segment(self, ltm_id: str, soft: bool = True) -> bool:
        return self.storage.delete_long_term_summary(ltm_id, soft=soft)

    # Key user operation: delete long-term knowledge manually.
    def delete_knowledge(self, knowledge_id: str, knowledge_type: str = "user") -> bool:
        if knowledge_type not in {"user", "assistant"}:
            raise ValueError("knowledge_type must be 'user' or 'assistant'")
        return self.storage.delete_knowledge(knowledge_id, knowledge_type=knowledge_type)
