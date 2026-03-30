import json
import os
import time
import asyncio
from pathlib import Path

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

from rag.query import get_disposal_advice_with_context

RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
EVAL_CASES = int(os.getenv("RAGAS_EVAL_CASES", "15"))
QUESTION_SLEEP_SEC = int(os.getenv("RAGAS_QUESTION_SLEEP_SEC", "45"))
RUN_TIMEOUT_SEC = int(os.getenv("RAGAS_RUN_TIMEOUT_SEC", "600"))
RUN_MAX_RETRIES = int(os.getenv("RAGAS_RUN_MAX_RETRIES", "4"))


class RateLimitedChatGroq(ChatGroq):
    """Sleeps before every call to stay within Groq free-tier 6 000 TPM limit."""
    SLEEP_SECONDS = int(os.getenv("RAGAS_LLM_CALL_SLEEP_SEC", "45"))

    def invoke(self, *args, **kwargs):
        time.sleep(self.SLEEP_SECONDS)
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        await asyncio.sleep(self.SLEEP_SECONDS)
        return await super().ainvoke(*args, **kwargs)

    def generate(self, *args, **kwargs):
        time.sleep(self.SLEEP_SECONDS)
        return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        await asyncio.sleep(self.SLEEP_SECONDS)
        return await super().agenerate(*args, **kwargs)


def _parse_question(question: str) -> tuple[list[str], str]:
    """Extract (materials, city) from 'How should I dispose of X in Y?' format."""
    lower_q = question.lower().strip(" ?")
    marker  = "how should i dispose of "
    if lower_q.startswith(marker) and " in " in lower_q:
        suffix = lower_q[len(marker):]
        material_part, city_part = suffix.rsplit(" in ", 1)
        return [material_part], city_part
    return [question], "unknown city"


def run_ragas_evaluation() -> dict:
    project_root      = Path(__file__).resolve().parent.parent
    ground_truth_path = project_root / "tests" / "rag_ground_truth.json"

    with ground_truth_path.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Legacy ragas.metrics uses the old column names
    questions:     list[str]       = []
    answers:       list[str]       = []
    contexts:      list[list[str]] = []
    ground_truths: list[str]       = []

    for i, item in enumerate(test_cases[:EVAL_CASES]):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"\n[{i+1}/{EVAL_CASES}] {question}", flush=True)

        parsed_materials, parsed_city = _parse_question(question)

        try:
            answer, retrieved_contexts = get_disposal_advice_with_context(
                materials=parsed_materials,
                city=parsed_city,
            )
        except Exception as e:
            print(f"  WARNING: Pipeline error - skipping: {e}", flush=True)
            continue

        if not answer or answer.startswith("Error:"):
            print(f"  WARNING: Pipeline error answer - skipping: {answer!r}", flush=True)
            continue

        print(f"  Answer:   {answer[:120]}...", flush=True)
        print(f"  Contexts: {len(retrieved_contexts)} chunks retrieved.", flush=True)
        print(f"  [Sleeping {QUESTION_SLEEP_SEC}s for Groq TPM reset...]", flush=True)
        time.sleep(QUESTION_SLEEP_SEC)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    if not questions:
        raise RuntimeError("No valid RAG responses generated. Check the RAG pipeline.")

    # Legacy ragas.metrics still uses the old column names
    hf_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    groq_judge = RateLimitedChatGroq(
        model=RAGAS_JUDGE_MODEL,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(groq_judge)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    answer_relevancy.strictness = 1

    # 70B judge calls can exceed 120s under quota pressure; keep workers=1 and use sane retries.
    run_config = RunConfig(max_workers=1, max_retries=RUN_MAX_RETRIES, timeout=RUN_TIMEOUT_SEC)

    scores = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    df = scores.to_pandas()

    # RAGAS v1.0 renames output columns regardless of input names:
    #   question  -> user_input
    #   answer    -> response
    #   contexts  -> retrieved_contexts
    #   ground_truth -> reference
    # Detect which naming convention the installed version uses
    q_col = "user_input" if "user_input" in df.columns else "question"
    a_col = "response"   if "response"   in df.columns else "answer"

    print("\n--- Per-question breakdown ---")
    print(df[[q_col, "faithfulness", "answer_relevancy"]].to_string(index=False))

    faithfulness_score = float(df["faithfulness"].mean())
    answer_relevancy_score = float(df["answer_relevancy"].mean())

    print(f"\nfaithfulness:      {faithfulness_score:.4f}  (target: >0.85)")
    print(f"answer_relevancy:  {answer_relevancy_score:.4f}  (target: >0.85)")

    bad_rows = df[df["faithfulness"] < 0.7]
    if not bad_rows.empty:
        print("\nWARNING: Low-faithfulness rows to investigate:")
        for _, row in bad_rows.iterrows():
            print(f"  Q: {row[q_col]}")
            print(f"  A: {row[a_col]}")
            print(f"  faithfulness: {row['faithfulness']:.2f}\n")

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
    }


def test_ragas_evaluation() -> None:
    results = run_ragas_evaluation()
    assert results["faithfulness"] >= 0.85, (
        f"faithfulness {results['faithfulness']:.4f} is below 0.85 KPI target"
    )
    assert results["answer_relevancy"] >= 0.85, (
        f"answer_relevancy {results['answer_relevancy']:.4f} is below 0.85 KPI target"
    )


if __name__ == "__main__":
    run_ragas_evaluation()
