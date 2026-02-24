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

    # Create Memoryos instance
    memoryos = Memoryos(
        user_id='travel_user_test',
        openai_api_key=api_key,
        openai_base_url=base_url,
        data_storage_path=data_path,
        assistant_id='travel_assistant',
        embedding_model_name=embedding_model,
        mid_term_capacity=1000,
        mid_term_heat_threshold=12.0,
        mid_term_similarity_threshold=0.7,
        short_term_capacity=2,
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
        ("Yes, I hope to deeply experience local life", "Deep travel is very meaningful! Would you like to try staying at a guesthouse or traditional inn?"),
        ("I enjoy photography, what are some good photo spots?", "Bamboo Grove, Fushimi Inari's thousands of torii gates are perfect for photography!"),
        ("I especially like photographing architecture and people", "You'll definitely love Kinkaku-ji's reflection and the geisha district streetscapes."),
        ("I don't like crowded places", "I recommend some early morning time slots, fewer tourists and great lighting."),
        ("I want to buy some traditional crafts as souvenirs", "Nishijin weaving items and Kiyomizu pottery tea sets have great collectible value."),
        ("Are there any seasonal experience activities?", "In October you can participate in momiji-gari (autumn leaf viewing) and hot spring bathing while viewing maples."),
        ("I want to try some local lifestyle experiences", "We can arrange early morning visits to fish markets to experience locals' rhythm."),
        ("For accommodation, I hope to experience different types", "We can arrange 2 nights at traditional ryokan, others at boutique guesthouses."),
        ("I'm also interested in Japanese flower arrangement", "Kyoto has many ikebana school experience classes, we can arrange one session."),
        ("Do you have shopping suggestions?", "I recommend some long-established shops, good quality and historical significance."),
        ("Overall, I hope this trip has rich cultural content", "Understood! I'll arrange a deep cultural experience journey for you.")
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

    print("\n📝 Phase 2: Add/delete long-term knowledge item...")
    knowledge_id = memoryos.user_long_term_memory.add_knowledge(
        "User prefers Japanese cultural travel and avoids crowded spots",
        knowledge_type="user"
    )
    print(f"  Added user knowledge id: {knowledge_id}")
    deleted_knowledge = memoryos.delete_long_term_knowledge(knowledge_id, knowledge_type="user")
    print(f"  Delete user knowledge: {deleted_knowledge}")

    print(f"\n🔥 Phase 3: Force triggering mid-term analysis...")
    memoryos.force_mid_term_analysis()

    print(f"\n⏳ Phase 4: Waiting for system synchronization...")
    time.sleep(2)

    print("\n📝 Phase 5: Manual delete operations (short/long/key)...")
    # 1) short-term manual delete by step
    short_term_items = memoryos.short_term_memory.get_all()
    short_delete_ok = False
    if short_term_items:
        step_to_delete = short_term_items[0].get("step")
        if step_to_delete is not None:
            short_delete_ok = memoryos.delete_short_term_memory(step_to_delete)
            print(f"  Delete short-term step={step_to_delete}: {short_delete_ok}")

    # 2) long-term summary manual delete
    long_summaries = memoryos.user_long_term_memory.list_long_term_summaries(limit=5)
    long_delete_ok = False
    if long_summaries:
        ltm_id = long_summaries[0]["ltm_id"]
        long_delete_ok = memoryos.delete_long_term_summary(ltm_id, soft=True)
        print(f"  Delete long-term summary {ltm_id}: {long_delete_ok}")

    # 3) key memory manual delete
    key_delete_ok = memoryos.delete_key_memory(manual_key_id, soft=True)
    print(f"  Delete key memory {manual_key_id}: {key_delete_ok}")

    print(f"\n🧠 Phase 6: Testing Memory System Query Response...")

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
        },
        {
            "query": "What are my hobbies and what kind of experiences am I interested in?",
            "expected_keywords": ["photography", "traditional", "cultural"],
            "description": "Testing hobby and interest recall"
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

    print("\n📝 Phase 7: Audit check...")
    events = memoryos.list_memory_events(limit=30)
    print(f"  Recent memory events: {len(events)}")
    has_short_delete = any(e.get("memory_type") == "short_term" and "delete" in e.get("action", "") for e in events)
    has_long_delete = any(e.get("memory_type") in ("long_term_summary", "user_knowledge") and "delete" in e.get("action", "") for e in events)
    has_key_delete = any(e.get("memory_type") == "key_memory" and "delete" in e.get("action", "") for e in events)

    delete_pass = short_delete_ok and long_delete_ok and key_delete_ok and has_short_delete and has_long_delete and has_key_delete
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
    print(f"Delete Ops Check: {delete_pass}")
    
    if success_rate >= 70 and delete_pass:
        print("\n🎉 EXCELLENT! Memory system performed very well!")
        return True
    elif success_rate >= 50 and delete_pass:
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
