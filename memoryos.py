import os
import json
import atexit
from concurrent.futures import ThreadPoolExecutor

# 修改为绝对导入
try:
    # 尝试相对导入（当作为包使用时）
    from .utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists, set_embedding_api_client
    from . import prompts
    from .storage_provider import ChromaStorageProvider
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .key_term import KeyTermMemory
    from .updater import Updater
    from .retriever import Retriever
except ImportError:
    # 回退到绝对导入（当作为独立模块使用时）
    from utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, ensure_directory_exists, set_embedding_api_client
    import prompts
    from storage_provider import ChromaStorageProvider
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory, compute_segment_heat # For H_THRESHOLD logic
    from long_term import LongTermMemory
    from key_term import KeyTermMemory
    from updater import Updater
    from retriever import Retriever

# Heat threshold for triggering profile/knowledge update from mid-term memory
H_PROFILE_UPDATE_THRESHOLD = 5.0 
DEFAULT_ASSISTANT_ID = "default_assistant_profile"

class Memoryos:
    def __init__(self, user_id: str, 
                 openai_api_key: str, 
                 data_storage_path: str,
                 openai_base_url: str = None, 
                 assistant_id: str = DEFAULT_ASSISTANT_ID, 
                 short_term_capacity=10,
                 mid_term_capacity=2000,
                 mid_term_page_merge_threshold=60,
                 mid_term_page_merge_keep_tail=20,
                 long_term_knowledge_capacity=100,
                 retrieval_queue_capacity=7,
                 mid_term_heat_threshold=H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold=0.6,
                 key_promotion_hit_rate=0.35,
                 key_promotion_min_hits=8,
                 key_promotion_heat=6.0,
                 llm_model="gpt-4o-mini",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_model_kwargs: dict = None,
                 use_embedding_api: bool = False,  # 新增：是否使用API调用embedding服务
                 ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.key_promotion_hit_rate = key_promotion_hit_rate
        self.key_promotion_min_hits = key_promotion_min_hits
        self.key_promotion_heat = key_promotion_heat
        self.embedding_model_name = embedding_model_name
        self.use_embedding_api = use_embedding_api  # 保存参数
        
        # Smart defaults for embedding_model_kwargs
        if embedding_model_kwargs is None:
            if not use_embedding_api and 'bge-m3' in self.embedding_model_name.lower():
                print("INFO: Detected bge-m3 model, defaulting embedding_model_kwargs to {'use_fp16': True}")
                self.embedding_model_kwargs = {'use_fp16': True}
            else:
                self.embedding_model_kwargs = {}
        else:
            self.embedding_model_kwargs = dict(embedding_model_kwargs)  # Ensure it's a mutable dict
        
        print(f"Initializing Memoryos for user '{self.user_id}' and assistant '{self.assistant_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")
        print(f"Using embedding model: {self.embedding_model_name} (use_api={self.use_embedding_api}) with kwargs: {self.embedding_model_kwargs}")

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)
        
        # 如果使用 API embedding，设置全局 embedding API client
        if self.use_embedding_api:
            set_embedding_api_client(self.client)
        
        # Centralized Storage Provider
        storage_path = os.path.join(self.data_storage_path, "chroma_storage")
        self.storage_provider = ChromaStorageProvider(
            path=storage_path, 
            user_id=self.user_id, 
            assistant_id=self.assistant_id
        )

        # Register save handler to be called on exit
        atexit.register(self.close)

        # Initialize Memory Modules with the shared storage provider
        self.short_term_memory = ShortTermMemory(
            storage_provider=self.storage_provider,
            max_capacity=short_term_capacity
        )
        self.mid_term_memory = MidTermMemory(
            storage_provider=self.storage_provider,
            user_id=self.user_id,
            client=self.client, 
            max_capacity=mid_term_capacity,
            page_merge_threshold=mid_term_page_merge_threshold,
            page_merge_keep_tail=mid_term_page_merge_keep_tail,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            llm_model=self.llm_model,
            use_embedding_api=self.use_embedding_api
        )
        self.user_long_term_memory = LongTermMemory(
            storage_provider=self.storage_provider,
            llm_interface=self.client,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            llm_model=self.llm_model,
            use_embedding_api=self.use_embedding_api
        )
        self.key_term_memory = KeyTermMemory(
            storage_provider=self.storage_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            use_embedding_api=self.use_embedding_api
        )

        # Initialize Memory Module for Assistant Knowledge
        self.assistant_long_term_memory = LongTermMemory(
            storage_provider=self.storage_provider,
            llm_interface=self.client,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs,
            llm_model=self.llm_model,
            use_embedding_api=self.use_embedding_api
        )

        # Initialize Orchestration Modules
        self.updater = Updater(
            short_term_memory=self.short_term_memory, 
            mid_term_memory=self.mid_term_memory, 
            long_term_memory=self.user_long_term_memory,
            client=self.client,
            topic_similarity_threshold=mid_term_similarity_threshold,
            llm_model=self.llm_model
        )
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            user_long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory,
            key_term_memory=self.key_term_memory,
            queue_capacity=retrieval_queue_capacity
        )
        
        self.mid_term_heat_threshold = mid_term_heat_threshold

    def close(self):
        """Saves all metadata to disk. Registered with atexit to be called on script termination."""
        print("Memoryos: Process is terminating. Saving all metadata to disk...")
        self.storage_provider.save_all_metadata()
        print("Memoryos: Metadata saved successfully.")

    def _trigger_profile_and_knowledge_update_if_needed(self):
        """
        Checks mid-term memory for hot segments and triggers profile/knowledge update if threshold is met.
        Adapted from main_memoybank.py's update_user_profile_from_top_segment.
        Enhanced with parallel LLM processing for better performance.
        """
        if not self.mid_term_memory.heap:
            return

        # Peek at the top of the heap (hottest segment)
        # MidTermMemory heap stores (-H_segment, sid)
        neg_heat, sid = self.mid_term_memory.heap[0] 
        current_heat = -neg_heat

        if current_heat >= self.mid_term_heat_threshold:
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                self.mid_term_memory.rebuild_heap() # Clean up if session is gone
                return

            # Get unanalyzed pages from this hot session
            unanalyzed_pages = [
                page for page in self.mid_term_memory.storage.get_pages_from_json_backup(sid)
                if not page.get("analyzed")
            ]

            if unanalyzed_pages:
                print(f"Memoryos: Mid-term session {sid} heat ({current_heat:.2f}) exceeded threshold. Analyzing {len(unanalyzed_pages)} pages for profile/knowledge update.")
                
                # Combine all unanalyzed page interactions into a single string for LLM
                conversation_str = "\n".join(
                    [f"User: {p.get('user_input', '')}\nAssistant: {p.get('agent_response', '')}" for p in unanalyzed_pages]
                )

                def task_update_profile():
                    print("Memoryos: Starting user profile update task...")
                    return self.user_long_term_memory.update_user_profile(self.user_id, conversation_str)

                def task_extract_knowledge():
                    print("Memoryos: Starting knowledge extraction task...")
                    # This function needs the raw conversation string from the hot pages
                    return self.user_long_term_memory.extract_knowledge_from_text(conversation_str)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_profile = executor.submit(task_update_profile)
                    future_knowledge = executor.submit(task_extract_knowledge)

                    try:
                        updated_profile = future_profile.result()
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        print(f"Error in parallel LLM processing: {e}")
                        return

                # The profile is already updated in memory by update_user_profile
                if updated_profile:
                    self.storage_provider.record_update_time("profile", get_timestamp())
                    print(f"Memoryos: User profile update recorded in memory for user {self.user_id}.")
                
                # Add extracted knowledge
                if knowledge_result:
                    user_knowledge = knowledge_result.get("private")
                    if user_knowledge:
                        # Ensure user_knowledge is a list before iterating
                        if isinstance(user_knowledge, str):
                            user_knowledge = [user_knowledge]
                        for item in user_knowledge:
                            self.user_long_term_memory.add_knowledge(item, "user")
                    
                    assistant_knowledge = knowledge_result.get("assistant_knowledge")
                    if assistant_knowledge:
                        # Ensure assistant_knowledge is a list before iterating
                        if isinstance(assistant_knowledge, str):
                            assistant_knowledge = [assistant_knowledge]
                        for item in assistant_knowledge:
                            self.assistant_long_term_memory.add_knowledge(item, "assistant")

                    self.storage_provider.record_update_time("knowledge", get_timestamp())

                # Mark pages as analyzed and reset session heat
                for page in unanalyzed_pages:
                    page["analyzed"] = True
                
                # Update the session metadata in storage
                session["N_visit"] = 0 
                # Keep L_interaction intact; it represents segment size, not analysis state.
                session["H_segment"] = compute_segment_heat(session)
                session["last_visit_time"] = get_timestamp()
                self.mid_term_memory.storage.update_mid_term_session_metadata(sid, session)

                self.mid_term_memory.rebuild_heap()
                print(f"Memoryos: Profile/Knowledge update for session {sid} complete. Heat reset.")
            else:
                print(f"Memoryos: Hot session {sid} has no unanalyzed pages. Skipping profile update.")
        else:
            # print(f"Memoryos: Top session {sid} heat ({current_heat:.2f}) below threshold. No profile update.")
            pass # No action if below threshold

    def _rollup_long_term_summary(self, updater_result: dict):
        if not updater_result:
            return
        # Prefer queue-window rollup (same length as STM capacity) when provided.
        text = updater_result.get("long_term_rollup_text", "").strip() or updater_result.get("input_text_for_summary", "").strip()
        step_start = updater_result.get("long_term_step_start", updater_result.get("step_start"))
        step_end = updater_result.get("long_term_step_end", updater_result.get("step_end"))
        if not text or step_start is None or step_end is None:
            return
        self.user_long_term_memory.add_summary_segment(
            text=text,
            step_start=step_start,
            step_end=step_end,
            source="short_term_rollup",
            version="v1"
        )

    def _auto_promote_key_memory(self):
        candidates = self.mid_term_memory.collect_key_memory_candidates(
            hit_rate_threshold=self.key_promotion_hit_rate,
            min_hits=self.key_promotion_min_hits,
            heat_threshold=self.key_promotion_heat,
            limit=3
        )
        for item in candidates:
            text = item.get("text", "").strip()
            if not text:
                continue
            self.key_term_memory.add_auto_key_memory(
                text=text,
                trace=item.get("trace"),
                confidence=item.get("confidence", 0.8),
                priority=item.get("priority", 0.85),
                category="goal",
                step=self.storage_provider.get_global_step()
            )

    def add_memory(self, user_input: str, agent_response: str, timestamp = None, meta_data = None):
        """
        Adds a new QA pair (memory) to the system.
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()
        step = self.storage_provider.increment_global_step()
        
        qa_pair = {
            "user_id": self.user_id, # Add user_id to qa_pair
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "step": step,
            "status": "active",
            "version": "v1"
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        self.storage_provider.record_memory_event("add", "short_term", f"qa_{step}")
        print(f"Memoryos: Added QA to short-term. User: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            print("Memoryos: Short-term memory full. Processing to mid-term.")
            updater_result = self.updater.process_short_term_to_mid_term()
            # Execute sequentially to avoid concurrent writes to shared metadata state.
            self._rollup_long_term_summary(updater_result)
            self._auto_promote_key_memory()
        
        # After any memory addition that might impact mid-term, check for profile updates
        self._trigger_profile_and_knowledge_update_if_needed()

    def get_response(self, query: str, relationship_with_user="friend", style_hint="", user_conversation_meta_data = None) -> str:
        """
        Generates a response to the user's query, incorporating memory and context.
        """
        print(f"Memoryos: Generating response for query: '{query[:50]}...'")

        # 1. Get short-term history first, then retrieve context in parallel
        short_term_history = self.short_term_memory.get_all()
        retrieval_results = self.retriever.retrieve_context(
            user_query=query,
            user_id=self.user_id,
            retrieved_short_term=short_term_history
            # Using default thresholds from Retriever class for now
        )
        retrieved_key_memory = retrieval_results["retrieved_key_memory"]
        retrieved_long_term = retrieval_results["retrieved_long_term"]
        retrieved_pages = retrieval_results["retrieved_pages"]
        retrieved_short_term = retrieval_results["retrieved_short_term"]
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]

        # 2. Format short-term history
        history_text = "\n".join([
            f"[step={qa.get('step', '')}] User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in retrieved_short_term
        ])

        # 3. Format retrieved key memory
        key_memory_manual = [m for m in retrieved_key_memory if m.get("source") == "manual"]
        key_memory_auto = [m for m in retrieved_key_memory if m.get("source") != "manual"]
        key_memory_lines = []
        for km in key_memory_manual + key_memory_auto:
            key_memory_lines.append(
                f"- ({km.get('source', 'manual')}, p={km.get('priority', 0):.2f}, step={km.get('step', '')}) {km.get('text', '')}"
            )
        key_memory_text = "【关键记忆】\n" + ("\n".join(key_memory_lines) if key_memory_lines else "- None")

        # 4. Format retrieved long-term summaries
        ltm_lines = []
        for seg in retrieved_long_term:
            ltm_lines.append(
                f"- [step {seg.get('step_start', '?')}~{seg.get('step_end', '?')}] {seg.get('text', '')}"
            )
        long_term_text = "【长期记忆】\n" + ("\n".join(ltm_lines) if ltm_lines else "- None")

        # 5. Format retrieved mid-term pages with dialogue chain info
        def _build_mid_term_page_text(page):
            page_id = page.get("page_id", "")
            session_id = page.get("session_id", "")
            meta_info = ""
            if page_id and session_id:
                full_page_info = self.storage_provider.get_page_full_info(page_id, session_id)
                if full_page_info:
                    meta_info = full_page_info.get("meta_info", "")
            page_text = f"User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}"
            if meta_info:
                page_text += f"\n Dialogue chain info: \n{meta_info}"
            return page_text

        retrieval_text_parts = []
        if retrieved_pages:
            # Parallelize page-text construction when many recalled pages are present.
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(retrieved_pages)))) as executor:
                retrieval_text_parts = list(executor.map(_build_mid_term_page_text, retrieved_pages))
        retrieval_text = "【中期记忆】\n" + ("\n\n".join(retrieval_text_parts) if retrieval_text_parts else "- None")
        short_term_text = "【短期记忆】\n" + (history_text if history_text else "- None")
        ordered_memory_text = f"{key_memory_text}\n\n{long_term_text}\n\n{retrieval_text}\n\n{short_term_text}"

        # 6. Get user profile
        user_profile_data = self.user_long_term_memory.get_user_profile(self.user_id)
        user_profile_text = json.dumps(user_profile_data, indent=2, ensure_ascii=False) if user_profile_data else "No detailed profile available yet."

        # 7. Format retrieved user knowledge
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = "\n【Relevant User Knowledge】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['text']}\n"
        
        background_context = f"【User Profile】\n{user_profile_text}\n{user_knowledge_background}"

        # 8. Format retrieved Assistant Knowledge
        assistant_knowledge_text_for_prompt = "【Assistant Knowledge Base】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['text']}\n"
        else:
            assistant_knowledge_text_for_prompt += "- No relevant assistant knowledge found for this query.\n"

        # 9. Format user_conversation_meta_data (if provided)
        meta_data_text_for_prompt = "【Current Conversation Metadata】\n"
        if user_conversation_meta_data:
            try:
                meta_data_text_for_prompt += json.dumps(user_conversation_meta_data, ensure_ascii=False, indent=2)
            except TypeError:
                meta_data_text_for_prompt += str(user_conversation_meta_data)
        else:
            meta_data_text_for_prompt += "None provided for this turn."

        # 10. Construct Prompts
        system_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT.format(
            relationship=relationship_with_user,
            assistant_knowledge_text=assistant_knowledge_text_for_prompt,
            meta_data_text=meta_data_text_for_prompt # Using meta_data_text placeholder for user_conversation_meta_data
        )
        
        user_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT.format(
            history_text=short_term_text,
            retrieval_text=ordered_memory_text,
            background=background_context,
            relationship=relationship_with_user,
            query=query
        )
        
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]

        # 11. Call LLM for response
        print("Memoryos: Calling LLM for final response generation...")
        # print("System Prompt:\n", system_prompt_text)
        # print("User Prompt:\n", user_prompt_text)
        response_content = self.client.chat_completion(
            model=self.llm_model, 
            messages=messages, 
            temperature=0.7, 
            max_tokens=1500 # As in original main
        )
        
        # 12. Add this interaction to memory
        self.add_memory(user_input=query, agent_response=response_content, timestamp=get_timestamp())
        
        return response_content

    # --- Helper/Maintenance methods (optional additions) ---
    def add_key_memory(self, text: str, category: str = "preference", priority: float = 0.95):
        return self.key_term_memory.add_manual_key_memory(
            text=text,
            category=category,
            priority=priority,
            step=self.storage_provider.get_global_step()
        )

    def delete_key_memory(self, key_id: str, soft: bool = True):
        return self.key_term_memory.delete_key_memory(key_id, soft=soft)

    def restore_key_memory(self, key_id: str):
        return self.key_term_memory.restore_key_memory(key_id)

    # Convenience wrappers for manual memory management in products/tools.
    def delete_short_term_memory(self, step: int):
        return self.delete_memory("short_term", str(step), soft=True)

    def restore_short_term_memory(self, step: int):
        return self.restore_memory("short_term", str(step))

    def delete_long_term_summary(self, ltm_id: str, soft: bool = True):
        return self.delete_memory("long_term_summary", ltm_id, soft=soft)

    def delete_long_term_knowledge(self, knowledge_id: str, knowledge_type: str = "user"):
        memory_type = "user_knowledge" if knowledge_type == "user" else "assistant_knowledge"
        return self.delete_memory(memory_type, knowledge_id, soft=False)

    def delete_mid_term_by_keywords(self, keywords, soft: bool = True):
        deleted_session_ids = self.mid_term_memory.delete_sessions_by_keywords(keywords, soft=soft)
        for sid in deleted_session_ids:
            self.storage_provider.record_memory_event(
                "delete_soft" if soft else "delete_hard",
                "mid_term_session",
                sid,
                detail={"keywords": keywords}
            )
        return deleted_session_ids

    def restore_mid_term_session(self, session_id: str):
        ok = self.mid_term_memory.restore_session(session_id)
        if ok:
            self.storage_provider.record_memory_event("restore", "mid_term_session", session_id)
        return ok

    def delete_memory(self, memory_type: str, memory_id: str, soft: bool = True):
        # Unified delete API for UI/client calls.
        if memory_type == "short_term":
            # `memory_id` for short-term is dialogue step.
            step = int(memory_id)
            ok = self.short_term_memory.delete_by_step(step, soft=soft)
            if ok:
                self.storage_provider.record_memory_event("delete_soft" if soft else "delete_hard", "short_term", f"qa_{step}")
            return ok
        if memory_type == "key_memory":
            return self.delete_key_memory(memory_id, soft=soft)
        if memory_type == "mid_term_session":
            ok = self.mid_term_memory.storage.delete_mid_term_session(memory_id, soft=soft)
            if ok:
                if soft and memory_id in self.mid_term_memory.sessions:
                    self.mid_term_memory.sessions[memory_id]["status"] = "deleted"
                elif not soft:
                    self.mid_term_memory.sessions.pop(memory_id, None)
                    # Keep local LFU state consistent after hard delete.
                    self.mid_term_memory.access_frequency.pop(memory_id, None)
                self.mid_term_memory.rebuild_heap()
                self.mid_term_memory.save()
                self.storage_provider.record_memory_event("delete_soft" if soft else "delete_hard", memory_type, memory_id)
            return ok
        if memory_type == "long_term":
            # Default long-term deletion target is summary segment for backward compatibility.
            memory_type = "long_term_summary"
        if memory_type == "long_term_summary":
            ok = self.user_long_term_memory.delete_summary_segment(memory_id, soft=soft)
            if ok:
                self.storage_provider.record_memory_event("delete_soft" if soft else "delete_hard", memory_type, memory_id)
            return ok
        if memory_type == "user_knowledge":
            ok = self.user_long_term_memory.delete_knowledge(memory_id, knowledge_type="user")
            if ok:
                self.storage_provider.record_memory_event("delete_hard", memory_type, memory_id)
            return ok
        if memory_type == "assistant_knowledge":
            ok = self.assistant_long_term_memory.delete_knowledge(memory_id, knowledge_type="assistant")
            if ok:
                self.storage_provider.record_memory_event("delete_hard", memory_type, memory_id)
            return ok
        raise ValueError(f"Unsupported memory_type: {memory_type}")

    def restore_memory(self, memory_type: str, memory_id: str):
        if memory_type == "short_term":
            step = int(memory_id)
            ok = self.short_term_memory.restore_by_step(step)
            if ok:
                self.storage_provider.record_memory_event("restore", "short_term", f"qa_{step}")
            return ok
        if memory_type == "key_memory":
            return self.restore_key_memory(memory_id)
        if memory_type == "mid_term_session":
            ok = self.mid_term_memory.restore_session(memory_id)
            if ok:
                self.storage_provider.record_memory_event("restore", memory_type, memory_id)
            return ok
        if memory_type == "long_term":
            memory_type = "long_term_summary"
        if memory_type == "long_term_summary":
            ok = self.storage_provider.restore_long_term_summary(memory_id)
            if ok:
                self.storage_provider.record_memory_event("restore", memory_type, memory_id)
            return ok
        raise ValueError(f"Unsupported memory_type: {memory_type}")

    def list_memory_events(self, limit: int = 50):
        return self.storage_provider.list_memory_events(limit=limit)

    def get_user_profile_summary(self) -> dict:
        """Retrieves the full user profile object."""
        profile = self.user_long_term_memory.get_user_profile(self.user_id)
        return profile or {}

    def get_assistant_knowledge_summary(self) -> list:
        return self.assistant_long_term_memory.get_assistant_knowledge()

    def force_mid_term_analysis(self):
        """Forces analysis of all unanalyzed pages in the hottest mid-term segment if heat is above 0.
           Useful for testing or manual triggering.
        """
        original_threshold = self.mid_term_heat_threshold
        self.mid_term_heat_threshold = 0.0 # Temporarily lower threshold
        print("Memoryos: Force-triggering mid-term analysis...")
        self._trigger_profile_and_knowledge_update_if_needed()
        self.mid_term_heat_threshold = original_threshold # Restore original threshold

    def __repr__(self):
        return f"<Memoryos user_id='{self.user_id}' assistant_id='{self.assistant_id}' data_path='{self.data_storage_path}'>" 
