from typing import Optional, List

try:
    from .utils import get_embedding, normalize_vector, generate_id, get_timestamp
    from .storage_provider import ChromaStorageProvider
except ImportError:
    from utils import get_embedding, normalize_vector, generate_id, get_timestamp
    from storage_provider import ChromaStorageProvider


class KeyTermMemory:
    def __init__(
        self,
        storage_provider: ChromaStorageProvider,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_model_kwargs: Optional[dict] = None,
        use_embedding_api: bool = False
    ):
        self.storage = storage_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs or {}
        self.use_embedding_api = use_embedding_api

    def _embed(self, text: str) -> List[float]:
        vec = get_embedding(
            text,
            model_name=self.embedding_model_name,
            use_api=self.use_embedding_api,
            **self.embedding_model_kwargs
        )
        return normalize_vector(vec).tolist()

    def add_manual_key_memory(
        self,
        text: str,
        category: str = "preference",
        priority: float = 0.95,
        step: Optional[int] = None,
        timestamp: Optional[str] = None,
        ttl_days: Optional[int] = None,
        version: str = "v1"
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text for key memory cannot be empty")

        key_id = generate_id("km")
        payload = {
            "key_id": key_id,
            "user_id": self.storage.user_id,
            "assistant_id": self.storage.assistant_id,
            "text": text.strip(),
            "category": category,
            "source": "manual",
            "priority": float(priority),
            "confidence": 1.0,
            "ttl_days": ttl_days,
            "step": step if step is not None else self.storage.get_global_step(),
            "timestamp": timestamp or get_timestamp(),
            "status": "active",
            "version": version,
            "trace": {"rule": "manual_input", "session_ids": [], "page_ids": []}
        }
        self.storage.add_key_memory(payload, self._embed(payload["text"]))
        self.storage.record_memory_event("add", "key_memory", key_id, {"source": "manual", "category": category})
        return key_id

    def add_auto_key_memory(
        self,
        text: str,
        trace: Optional[dict] = None,
        confidence: float = 0.8,
        priority: float = 0.85,
        category: str = "goal",
        step: Optional[int] = None,
        timestamp: Optional[str] = None,
        version: str = "v1"
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text for key memory cannot be empty")

        key_id = generate_id("km")
        payload = {
            "key_id": key_id,
            "user_id": self.storage.user_id,
            "assistant_id": self.storage.assistant_id,
            "text": text.strip(),
            "category": category,
            "source": "auto",
            "priority": float(priority),
            "confidence": float(confidence),
            "ttl_days": None,
            "step": step if step is not None else self.storage.get_global_step(),
            "timestamp": timestamp or get_timestamp(),
            "status": "active",
            "version": version,
            "trace": trace or {"rule": "auto_extract", "session_ids": [], "page_ids": []}
        }
        self.storage.add_key_memory(payload, self._embed(payload["text"]))
        self.storage.record_memory_event("add", "key_memory", key_id, {"source": "auto", "trace": payload["trace"]})
        return key_id

    def search_key_memory(self, query: str, top_k: int = 5, threshold: float = 0.1) -> List[dict]:
        if not query.strip():
            return []
        query_embedding = self._embed(query)
        results = self.storage.search_key_memories(query_embedding, top_k=top_k, status="active")
        return [item for item in results if item.get("similarity", 0.0) >= threshold]

    def delete_key_memory(self, key_id: str, soft: bool = True) -> bool:
        ok = self.storage.delete_key_memory(key_id, soft=soft)
        if ok:
            self.storage.record_memory_event("delete_soft" if soft else "delete_hard", "key_memory", key_id)
        return ok

    def restore_key_memory(self, key_id: str) -> bool:
        ok = self.storage.restore_key_memory(key_id)
        if ok:
            self.storage.record_memory_event("restore", "key_memory", key_id)
        return ok

    def list_key_memory(self, status: str = "active", source: Optional[str] = None, limit: int = 100) -> List[dict]:
        return self.storage.list_key_memories(status=status, source=source, limit=limit)
