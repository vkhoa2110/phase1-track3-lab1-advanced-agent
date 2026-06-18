from __future__ import annotations
import json
import os
import re
from urllib import request
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
_RUNTIME_TOKEN_USAGE: list[int] = []

def base_qid(qid: str) -> str:
    return qid.split("_", 1)[0]

def failure_mode_for_qid(qid: str) -> str:
    return FAILURE_MODE_BY_QID.get(base_qid(qid), "wrong_final_answer")

def runtime_mode() -> str:
    mode = os.getenv("REFLEXION_RUNTIME", "mock").strip().lower()
    if mode in {"openai-compatible", "openai_compatible"}:
        return "openai_compatible"
    return mode

def set_runtime_mode(mode: str) -> None:
    normalized = mode.strip().lower()
    if normalized in {"openai-compatible", "openai_compatible"}:
        normalized = "openai_compatible"
    if normalized not in {"mock", "ollama", "openai", "openai_compatible"}:
        raise ValueError("runtime must be one of: mock, ollama, openai, openai_compatible")
    os.environ["REFLEXION_RUNTIME"] = normalized

def consume_runtime_token_usage() -> int:
    total = sum(_RUNTIME_TOKEN_USAGE)
    _RUNTIME_TOKEN_USAGE.clear()
    return total

def _record_token_usage(payload: dict) -> None:
    usage = payload.get("usage") or {}
    total = usage.get("total_tokens")
    if total is None:
        total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    if not total:
        total = (payload.get("prompt_eval_count") or 0) + (payload.get("eval_count") or 0)
    if total:
        _RUNTIME_TOKEN_USAGE.append(int(total))

def _model_name() -> str:
    model = (
        os.getenv("REFLEXION_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OLLAMA_MODEL")
    )
    if not model:
        raise RuntimeError("Set REFLEXION_LLM_MODEL before using a non-mock runtime.")
    return model

def _context_text(example: QAExample) -> str:
    return "\n".join(f"- {chunk.title}: {chunk.text}" for chunk in example.context)

def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    timeout = float(os.getenv("REFLEXION_LLM_TIMEOUT_SECONDS", "60"))
    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    _record_token_usage(result)
    return result

def _chat(messages: list[dict[str, str]]) -> str:
    mode = runtime_mode()
    model = _model_name()
    if mode == "ollama":
        base_url = os.getenv("REFLEXION_LLM_BASE_URL", "http://localhost:11434").rstrip("/")
        result = _post_json(
            f"{base_url}/api/chat",
            {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}},
        )
        return result["message"]["content"].strip()

    if mode in {"openai", "openai_compatible"}:
        base_url = os.getenv("REFLEXION_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        api_key = os.getenv("REFLEXION_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if mode == "openai" and not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or REFLEXION_LLM_API_KEY for REFLEXION_RUNTIME=openai.")
        result = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            {"model": model, "messages": messages, "temperature": 0},
            headers=headers,
        )
        return result["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"Unsupported runtime: {mode}")

def _json_object_from_text(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

def _llm_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    memory = "\n".join(reflection_memory) if reflection_memory else "None"
    return _chat(
        [
            {"role": "system", "content": ACTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Agent type: {agent_type}\n"
                    f"Attempt: {attempt_id}\n"
                    f"Question: {example.question}\n\n"
                    f"Context:\n{_context_text(example)}\n\n"
                    f"Reflection memory:\n{memory}"
                ),
            },
        ]
    )

def _llm_evaluator(example: QAExample, answer: str) -> JudgeResult:
    content = _chat(
        [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {example.question}\n"
                    f"Gold answer: {example.gold_answer}\n"
                    f"Predicted answer: {answer}"
                ),
            },
        ]
    )
    return JudgeResult.model_validate(_json_object_from_text(content))

def _llm_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    content = _chat(
        [
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Attempt id: {attempt_id}\n"
                    f"Question: {example.question}\n\n"
                    f"Context:\n{_context_text(example)}\n\n"
                    f"Evaluator reason: {judge.reason}\n"
                    f"Missing evidence: {judge.missing_evidence}\n"
                    f"Spurious claims: {judge.spurious_claims}"
                ),
            },
        ]
    )
    return ReflectionEntry.model_validate(_json_object_from_text(content))

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if runtime_mode() != "mock":
        return _llm_actor_answer(example, attempt_id, agent_type, reflection_memory)
    qid = base_qid(example.qid)
    if qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[qid]
    return example.gold_answer

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if runtime_mode() != "mock":
        return _llm_evaluator(example, answer)
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if runtime_mode() != "mock":
        return _llm_reflector(example, attempt_id, judge)
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if base_qid(example.qid) == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
