#!/usr/bin/env python3
"""
Comprehensive PMemo integration test (real API only).
This script validates:
1) Memory write + retrieval
2) Manual delete operations for short-term / long-term / key memory
"""

import sys
import os
import json
import time
sys.path.append('.')

from memoryos import Memoryos


def _must_get_env(name: str, fallback: str = "") -> str:
    value = os.getenv(name, "").strip()
    if not value:
        value = fallback.strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main():
    print("=" * 60)
    print("🚀 MemoryOS PMemo Comprehensive Real-API Test")
    print("=" * 60)

    # Real API configuration (no fake test logic).
    # Prefer env vars; fallback keeps this script runnable in local IDE setups.
    api_key = _must_get_env(
        "OPENAI_API_KEY",
        fallback="sk-hfhbxjmiwwthygrlpehackesymdqjtjlvdiksvqlvjompjys"
    )
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    llm_model = os.getenv("MEMORYOS_LLM_MODEL", "deepseek-ai/DeepSeek-V3")
    embedding_model = os.getenv("MEMORYOS_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
    data_path = os.getenv("MEMORYOS_TEST_DATA_PATH", "./tmp_chroma_schema_check")
    test_user_id = os.getenv("MEMORYOS_TEST_USER_ID", f"travel_user_test_{int(time.time())}")
    test_assistant_id = os.getenv("MEMORYOS_TEST_ASSISTANT_ID", "travel_assistant")

    # Create Memoryos instance
    memoryos = Memoryos(
        user_id=test_user_id,
        openai_api_key=api_key,
        openai_base_url=base_url,
        data_storage_path=data_path,
        assistant_id=test_assistant_id,
        embedding_model_name=embedding_model,
        mid_term_capacity=1000,
        mid_term_page_merge_threshold=4,
        mid_term_page_merge_keep_tail=2,
        # Keep this high in integration test to avoid expensive profile-analysis branch.
        mid_term_heat_threshold=999.0,
        mid_term_similarity_threshold=0.2,
        short_term_capacity=4,
        llm_model=llm_model,
        use_embedding_api=True
    )

    print("📝 Phase 0: Add manual key memory for deletion test...")
    manual_key_id = memoryos.add_key_memory("硬约束：预算不超过2万元，优先文化体验", category="constraint", priority=0.99)
    print(f"  Added manual key memory: {manual_key_id}")

    print("📝 Phase 1: Adding conversation rounds...")

    conversations = [
        ("Hello, I want to plan a trip", "Hello! I'd be happy to help you plan your trip. Where would you like to travel?"),
        ("My name is Emily, I'm 28 years old, and I'm a graphic designer", "Nice to meet you, Emily! As a graphic designer, you must have great aesthetic taste."),
        ("I prefer artistic and cultural travel destinations", "Artistic places are very charming! Do you prefer historical culture or modern art?"),
        ("I prefer historical culture, ancient architecture and museums", "Historical culture is very enriching! Have you considered European or Asian ancient cities?"),
        ("I want to go to Japan, especially Kyoto and Nara", "Japan's ancient capitals are beautiful! Kyoto's temples and Nara's deer are very famous."),
        ("I'm planning to go in October for about 7-10 days", "October is the best season to visit Japan! The autumn foliage season is beautiful."),
        ("My budget is around 15,000-20,000 yuan", "That's a reasonable budget! We can arrange a very nice itinerary."),
        ("I prefer niche places, don't want to go to overly commercialized spots", "I understand! You prefer experiencing local culture rather than tourist hotspots."),
        ("I enjoy photography, what are some good photo spots?", "Bamboo Grove and old streets are perfect for photography."),
        ("I don't like crowded places", "I recommend early morning time slots with fewer tourists.")
    ]
    
    # Add conversations
    for i, (user_input, agent_response) in enumerate(conversations, 1):
        print(f"  [{i:2d}/{len(conversations)}] Adding conversation: {user_input[:40]}...")
        memoryos.add_memory(user_input, agent_response)
        
        # Display status every 10 rounds
        if i % 10 == 0:
            sessions = memoryos.mid_term_memory.sessions
            if sessions:
                max_heat = max(session.get('H_segment', 0) for session in sessions.values())
                print(f"    Current max heat: {max_heat:.2f}")

    merged_triggered = any(
        s.get("compressed_at") is not None
        for s in memoryos.mid_term_memory.sessions.values()
    )
    print(f"  Mid-term page-merge/compress triggered: {merged_triggered}")

    print("\n📝 Phase 2: Add/delete long-term knowledge item...")
    knowledge_id = memoryos.user_long_term_memory.add_knowledge(
        "User prefers Japanese cultural travel and avoids crowded spots",
        knowledge_type="user"
    )
    print(f"  Added user knowledge id: {knowledge_id}")
    deleted_knowledge = memoryos.delete_long_term_knowledge(knowledge_id, knowledge_type="user")
    print(f"  Delete user knowledge: {deleted_knowledge}")

    print(f"\n⏳ Phase 3: Waiting for system synchronization...")
    time.sleep(2)

    print("\n📝 Phase 4: Manual delete operations (short/long/key)...")
    # 1) short-term manual delete by step
    short_term_items = memoryos.short_term_memory.get_all()
    short_delete_ok = False
    short_restore_ok = False
    if short_term_items:
        step_to_delete = short_term_items[0].get("step")
        if step_to_delete is not None:
            short_delete_ok = memoryos.delete_short_term_memory(step_to_delete)
            print(f"  Delete short-term step={step_to_delete}: {short_delete_ok}")
            short_restore_ok = memoryos.restore_short_term_memory(step_to_delete)
            print(f"  Restore short-term step={step_to_delete}: {short_restore_ok}")

    # 2) long-term summary manual delete
    long_summaries = memoryos.user_long_term_memory.list_long_term_summaries(limit=5)
    long_delete_ok = False
    long_restore_ok = False
    ltm_window_ok = any(seg.get("text", "").count("User:") == memoryos.short_term_memory.max_capacity for seg in long_summaries)
    print(f"  Long-term rollup window-size check: {ltm_window_ok}")
    if long_summaries:
        ltm_id = long_summaries[0]["ltm_id"]
        long_delete_ok = memoryos.delete_long_term_summary(ltm_id, soft=True)
        print(f"  Delete long-term summary {ltm_id}: {long_delete_ok}")
        long_restore_ok = memoryos.restore_memory("long_term_summary", ltm_id)
        print(f"  Restore long-term summary {ltm_id}: {long_restore_ok}")

    # 3) key memory manual delete
    key_delete_ok = memoryos.delete_key_memory(manual_key_id, soft=True)
    key_restore_ok = memoryos.restore_key_memory(manual_key_id)
    print(f"  Delete key memory {manual_key_id}: {key_delete_ok}")
    print(f"  Restore key memory {manual_key_id}: {key_restore_ok}")

    # 4) mid-term segment delete/restore by keyword
    mid_delete_ok = False
    mid_restore_ok = False
    delete_keywords = []
    session_for_restore = None
    for sid, session in memoryos.mid_term_memory.sessions.items():
        if session.get("status", "active") == "active":
            kws = session.get("summary_keywords") or []
            if kws:
                delete_keywords = [str(kws[0])]
                session_for_restore = sid
                break
            summary = (session.get("summary") or "").strip()
            if summary:
                delete_keywords = [summary.split()[0]]
                session_for_restore = sid
                break
    if delete_keywords:
        deleted_sessions = memoryos.delete_mid_term_by_keywords(delete_keywords, soft=True)
        mid_delete_ok = len(deleted_sessions) > 0
        print(f"  Delete mid-term by keywords={delete_keywords}: {deleted_sessions}")
        if session_for_restore:
            mid_restore_ok = memoryos.restore_mid_term_session(session_for_restore)
            print(f"  Restore mid-term session={session_for_restore}: {mid_restore_ok}")

    print(f"\n🧠 Phase 5: Testing Memory System Query Response...")

    test_queries = [
        {
            "query": "What's my name and what's my profession?",
            "expected_keywords": ["Emily", "graphic designer", "designer"],
            "description": "Testing basic user information recall"
        },
        {
            "query": "Where do I want to travel and what are my preferences?",
            "expected_keywords": ["Japan", "Kyoto", "Nara", "historical", "culture"],
            "description": "Testing travel destination and preference recall"
        }
    ]
    
    print("\n" + "="*60)
    print("🔍 MEMORY SYSTEM QUERY TESTING")
    print("="*60)
    
    total_score = 0
    max_score = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}: {test_case['description']}")
        print(f"Question: {test_case['query']}")
        print("-" * 50)
        
        try:
            # Get response from memory system
            response = memoryos.get_response(test_case['query'])
            print(f"System Response: {response}")
            
            # Check if expected keywords are in the response
            response_lower = response.lower()
            found_keywords = []
            missing_keywords = []
            
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            # Calculate score for this query
            keyword_score = len(found_keywords) / len(test_case['expected_keywords'])
            
            print(f"\n✅ Found keywords: {found_keywords}")
            if missing_keywords:
                print(f"❌ Missing keywords: {missing_keywords}")
            
            print(f"🎯 Keyword match rate: {keyword_score:.1%}")
            
            # Determine if this test passed (>50% keyword match)
            if keyword_score >= 0.5:
                print(f"✅ Test {i}: PASSED")
                total_score += 1
            else:
                print(f"❌ Test {i}: FAILED")
                
        except Exception as e:
            print(f"❌ Error during query {i}: {e}")
            print(f"❌ Test {i}: FAILED")

    print("\n📝 Phase 6: Audit check...")
    events = memoryos.list_memory_events(limit=30)
    print(f"  Recent memory events: {len(events)}")
    has_short_delete = any(e.get("memory_type") == "short_term" and "delete" in e.get("action", "") for e in events)
    has_short_restore = any(e.get("memory_type") == "short_term" and e.get("action") == "restore" for e in events)
    has_long_delete = any(e.get("memory_type") in ("long_term_summary", "user_knowledge") and "delete" in e.get("action", "") for e in events)
    has_long_restore = any(e.get("memory_type") == "long_term_summary" and e.get("action") == "restore" for e in events)
    has_key_delete = any(e.get("memory_type") == "key_memory" and "delete" in e.get("action", "") for e in events)
    has_key_restore = any(e.get("memory_type") == "key_memory" and e.get("action") == "restore" for e in events)
    has_mid_delete = any(e.get("memory_type") == "mid_term_session" and "delete" in e.get("action", "") for e in events)
    has_mid_restore = any(e.get("memory_type") == "mid_term_session" and e.get("action") == "restore" for e in events)

    delete_pass = all([
        short_delete_ok, short_restore_ok, long_delete_ok, long_restore_ok,
        key_delete_ok, key_restore_ok, mid_delete_ok, mid_restore_ok,
        has_short_delete, has_short_restore, has_long_delete, has_long_restore,
        has_key_delete, has_key_restore, has_mid_delete, has_mid_restore
    ])
    print(f"  Delete operation check: {delete_pass}")
    
    # Final results
    print("\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    
    success_rate = (total_score / max_score) * 100
    print(f"Passed Tests: {total_score}/{max_score}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Conversations Added: {len(conversations)}")
    print(f"Test Theme: Japan Travel Planning")
    print(f"User Profile: Emily, 28-year-old graphic designer, loves cultural travel and photography")
    print(f"Mid-term Merge Check: {merged_triggered}")
    print(f"Long-term Window Check: {ltm_window_ok}")
    print(f"Delete/Rollback Ops Check: {delete_pass}")
    
    if success_rate >= 70 and delete_pass and merged_triggered and ltm_window_ok:
        print("\n🎉 EXCELLENT! Memory system performed very well!")
        return True
    elif success_rate >= 50 and delete_pass and merged_triggered and ltm_window_ok:
        print("\n👍 GOOD! Memory system performed adequately!")
        return True
    else:
        print("\n😞 NEEDS IMPROVEMENT! Memory system needs optimization!")
        return False

if __name__ == "__main__":
    try:
        success = main()
    except Exception as e:
        print(f"\n❌ Test bootstrap failed: {e}")
        success = False
    if success:
        print("\n🎊 Congratulations! MemoryOS Travel Planning Memory Test Completed Successfully!")
    else:
        print("\n🔧 Memory system needs further optimization.")
