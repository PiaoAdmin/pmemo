import json
import os
import sys
from datetime import datetime

# Allow importing project-level Memoryos when running from eval/ directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(ROOT_DIR)
if WORKSPACE_DIR not in sys.path:
    sys.path.append(WORKSPACE_DIR)

from memoryP.memoryos import Memoryos
from config import API_KEY, BASE_URL, LLM_MODEL, EMBEDDING_MODEL, USE_EMBEDDING_API


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def process_conversation(conversation_data):
    """
    Convert LoCoMo conversation sessions into user/assistant QA pairs.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]

    session_keys = [k for k in conversation_data.keys() if k.startswith("session_") and not k.endswith("_date_time")]

    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")

        for dialog in conversation_data[session_key]:
            speaker = dialog.get("speaker", "")
            text = dialog.get("text", "")
            if dialog.get("blip_caption"):
                text = f"{text} (image description: {dialog['blip_caption']})"

            if speaker == speaker_a:
                processed.append(
                    {
                        "user_input": text,
                        "agent_response": "",
                        "timestamp": timestamp,
                    }
                )
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append(
                        {
                            "user_input": "",
                            "agent_response": text,
                            "timestamp": timestamp,
                        }
                    )
    return processed


def main():
    print("开始处理 locomo10 数据集（基于 Memoryos/ChromaDB）...")

    max_samples = int(os.getenv("LOCOMO_MAX_SAMPLES", "0"))
    max_qas_per_sample = int(os.getenv("LOCOMO_MAX_QAS_PER_SAMPLE", "0"))
    max_dialogs_per_sample = int(os.getenv("LOCOMO_MAX_DIALOGS_PER_SAMPLE", "0"))
    output_file = os.getenv("LOCOMO_OUTPUT_FILE", "all_loco_results.json")

    try:
        with open("locomo10.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print("错误：找不到 locomo10.json 文件，请确保文件在 eval 目录下")
        return
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return

    if max_samples > 0:
        dataset = dataset[:max_samples]
        print(f"快速模式：仅处理前 {len(dataset)} 个样本")

    # Use run-tagged folder to avoid interfering with previous test runs.
    run_tag = os.getenv("LOCOMO_RUN_TAG", datetime.now().strftime("%Y%m%d_%H%M%S"))
    chroma_root = os.path.join("mem_tmp_loco_final_chroma", run_tag)
    os.makedirs(chroma_root, exist_ok=True)

    results = []
    total_samples = len(dataset)

    for idx, sample in enumerate(dataset):
        sample_id = sample.get("sample_id", f"sample_{idx}")
        print(f"正在处理样本 {idx + 1}/{total_samples}: {sample_id}")

        conversation_data = sample.get("conversation", {})
        qa_pairs = sample.get("qa", [])
        processed_dialogs = process_conversation(conversation_data)

        if max_dialogs_per_sample > 0:
            processed_dialogs = processed_dialogs[:max_dialogs_per_sample]
            print(f"  快速模式：仅注入前 {len(processed_dialogs)} 条对话")

        if max_qas_per_sample > 0:
            qa_pairs = qa_pairs[:max_qas_per_sample]
            print(f"  快速模式：仅评测前 {len(qa_pairs)} 个问答")

        if not processed_dialogs:
            print(f"  样本 {sample_id} 没有有效对话，跳过")
            continue

        speaker_a = conversation_data.get("speaker_a", "user")
        speaker_b = conversation_data.get("speaker_b", "assistant")

        memoryos = Memoryos(
            user_id=sample_id,
            assistant_id=f"locomo_{speaker_b}",
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=os.path.join(chroma_root, sample_id),
            short_term_capacity=7,
            mid_term_capacity=200,
            mid_term_similarity_threshold=0.6,
            llm_model=LLM_MODEL,
            embedding_model_name=EMBEDDING_MODEL,
            use_embedding_api=USE_EMBEDDING_API,
            # Keep high in eval to avoid very expensive auto-analysis branch.
            mid_term_heat_threshold=999.0,
        )

        try:
            # Build memory from dialogues.
            for dialog in processed_dialogs:
                memoryos.add_memory(
                    user_input=dialog.get("user_input", ""),
                    agent_response=dialog.get("agent_response", ""),
                    timestamp=dialog.get("timestamp", "") or get_timestamp(),
                )

            # QA evaluation
            qa_count = len(qa_pairs)
            for qa_idx, qa in enumerate(qa_pairs):
                print(f"  处理问答 {qa_idx + 1}/{qa_count}")
                question = qa.get("question", "")
                original_answer = qa.get("answer", "") or qa.get("adversarial_answer", "")
                category = qa.get("category")
                evidence = qa.get("evidence", "")

                meta_data = {
                    "sample_id": sample_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "category": category,
                    "evidence": evidence,
                }

                try:
                    system_answer = memoryos.get_response(
                        question,
                        relationship_with_user="friend",
                        user_conversation_meta_data=meta_data,
                    )
                except Exception as e:
                    system_answer = f"[ERROR] {e}"

                results.append(
                    {
                        "sample_id": sample_id,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "question": question,
                        "system_answer": system_answer,
                        "original_answer": original_answer,
                        "category": category,
                        "evidence": evidence,
                        "timestamp": get_timestamp(),
                    }
                )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"样本 {idx + 1} 处理完成，结果已保存到 {output_file}")
        finally:
            memoryos.close()

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"全部处理完成，结果已保存到 {output_file}")
    except Exception as e:
        print(f"最终保存结果时出错：{e}")


if __name__ == "__main__":
    main()
