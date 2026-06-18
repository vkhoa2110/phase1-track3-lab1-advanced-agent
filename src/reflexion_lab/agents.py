from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
from .mock_runtime import actor_answer, consume_runtime_token_usage, evaluator, failure_mode_for_qid, reflector
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

def _context_text(example: QAExample) -> str:
    return "\n".join(f"{chunk.title}: {chunk.text}" for chunk in example.context)

def _estimate_tokens(*parts: object) -> int:
    text = " ".join(str(part) for part in parts if part)
    # A practical model-agnostic estimate for mock mode and providers that do
    # not return usage metadata. Real LLM integrations can replace this value.
    return max(1, round(len(text) / 4))

def _reflection_memory_text(reflection: ReflectionEntry) -> str:
    return (
        f"Attempt {reflection.attempt_id} failed: {reflection.failure_reason} "
        f"Lesson: {reflection.lesson} Next strategy: {reflection.next_strategy}"
    )

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        context_text = _context_text(example)
        for attempt_id in range(1, self.max_attempts + 1):
            consume_runtime_token_usage()
            attempt_started = perf_counter()
            memory_before_attempt = "\n".join(reflection_memory)
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            reflection: ReflectionEntry | None = None

            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(_reflection_memory_text(reflection))

            estimated_tokens = _estimate_tokens(
                ACTOR_SYSTEM,
                EVALUATOR_SYSTEM,
                example.question,
                example.gold_answer,
                context_text,
                memory_before_attempt,
                answer,
                judge.reason,
                judge.missing_evidence,
                judge.spurious_claims,
            )
            if reflection is not None:
                estimated_tokens += _estimate_tokens(
                    REFLECTOR_SYSTEM,
                    example.question,
                    context_text,
                    answer,
                    judge.reason,
                    reflection.lesson,
                    reflection.next_strategy,
                )
            token_estimate = consume_runtime_token_usage() or estimated_tokens
            latency_ms = max(1, round((perf_counter() - attempt_started) * 1000))
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = answer
            final_score = judge.score
            traces.append(trace)
            if judge.score == 1:
                break
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else failure_mode_for_qid(example.qid)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
