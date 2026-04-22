#!/usr/bin/env python3
"""Simple Ollama-based inference runner for Long Horizon Memory.

This script calls a local Ollama model (default: llama3.2:1b) to decide
append/rewrite/noop actions and prints step-level rewards for quick sanity checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Optional

import requests

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    from server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from .models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from .server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from long_horizon_memory.server.long_horizon_memory_environment import LongHorizonMemoryEnvironment


SYSTEM_PROMPT = """You are a memory compression agent.
Return JSON only with one of these shapes:
{"operation":"append"}
{"operation":"noop"}
{"operation":"rewrite","rewrite_memory":"..."}

Guidance:
- Behave in two phases:
    1) Gather phase: append likely factual messages; do not overuse noop early.
    2) Compress phase: once memory has enough content/noise, rewrite to distill key facts.
- If new_message contains fact-like relation statements, prefer append.
- If new_message is clearly off-topic/noisy, prefer noop.
- Use rewrite when memory has grown or contains noise; keep only answer-useful facts.
- Noop is allowed, but do not ignore too many potentially useful facts.
- Never output markdown or explanations.
"""


def maybe_strip_code_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        value = value.replace("json", "", 1).strip()
    return value


def compress_memory(memory: str, max_lines: int = 8, max_tokens: int = 110) -> str:
    if not memory.strip():
        return ""

    lines = [line.strip() for line in memory.splitlines() if line.strip()]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    words = " ".join(lines).split()
    if len(words) > max_tokens:
        words = words[-max_tokens:]
    return " ".join(words)


def compress_memory_fact_preserving(memory: str, max_lines: int = 8, max_tokens: int = 110) -> str:
    """Compress memory while prioritizing fact-like relation lines."""
    lines = [line.strip() for line in memory.splitlines() if line.strip()]
    if not lines:
        return ""

    fact_lines = [line for line in lines if strict_fact_signal(line)]
    chosen = fact_lines[:max_lines] if fact_lines else lines[-max_lines:]
    text = "\n".join(chosen)
    words = text.split()
    if len(words) > max_tokens:
        words = words[-max_tokens:]
        text = " ".join(words)
    return text


def heuristic_action(obs: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    text = (obs.new_message or "").lower()

    likely_relevant = [
        "head office",
        "founded",
        "born",
        "capital",
        "headquartered",
        "country",
        "city",
        "group",
        "family",
    ]
    likely_noise = [
        "planning",
        "kitchen",
        "weekend",
        "show",
        "bought",
        "renovate",
    ]

    if any(k in text for k in likely_relevant):
        return LongHorizonMemoryAction(operation="append")

    if any(k in text for k in likely_noise):
        return LongHorizonMemoryAction(operation="noop")

    if obs.memory_count > 130:
        return LongHorizonMemoryAction(
            operation="rewrite",
            rewrite_memory=compress_memory(obs.memory),
        )

    return LongHorizonMemoryAction(operation="noop")


def relevance_score(text: str) -> float:
    message = (text or "").lower()
    if not message.strip():
        return 0.0

    positive_patterns = [
        r"\bhead office\b",
        r"\bheadquartered\b",
        r"\bwas born in\b",
        r"\bis (in|from)\b",
        r"\bfounded\b",
        r"\bcapital\b",
        r"\bfamily\b",
        r"\bgroup\b",
        r"\bcompany\b",
        r"\binvolved in\b",
    ]
    negative_patterns = [
        r"\bi am planning\b",
        r"\bnext month\b",
        r"\bweekend\b",
        r"\bshow\b",
        r"\bbought\b",
        r"\brenovate\b",
        r"\bcafe\b",
        r"\bmarket\b",
        r"\bsports channel\b",
        r"\bferry tour\b",
        r"\bstation\b",
        r"\bmenu\b",
        r"\bdelays?\b",
    ]

    pos = sum(1 for pattern in positive_patterns if re.search(pattern, message))
    neg = sum(1 for pattern in negative_patterns if re.search(pattern, message))
    raw = 0.55 * pos - 0.6 * neg

    # Bias against generic narrative statements commonly used as distractors.
    if message.startswith(("a ", "some ")):
        raw -= 0.25

    return max(-1.0, min(1.0, raw))


def strict_fact_signal(text: str) -> bool:
    raw = (text or "").strip()
    lower = raw.lower()
    if not raw:
        return False

    # Suppress obvious conversational/noise framing.
    if any(
        cue in lower
        for cue in [
            "i am ",
            "i'm ",
            "my ",
            "next month",
            "weekend",
            "planning",
            "bought",
            "show",
        ]
    ):
        return False

    # Strong factual relation templates common in these episodes.
    templates = [
        r"^[A-Z][\w\-\s&']+ was born in [A-Z][\w\-\s&']+\.?$",
        r"^[A-Z][\w\-\s&']+ is (in|from) [A-Z][\w\-\s&']+\.?$",
        r"^[A-Z][\w\-\s&']+ was a [a-z][\w\-\s,']+\.?$",
        r"^[A-Z][\w\-\s&']+ (is|was) (a|an) [a-z][\w\-\s,']+\.?$",
        r"^[A-Z][\w\-\s&']+ (is|was) (headquartered|based) in [A-Z][\w\-\s&']+\.?$",
        r"^[A-Z][\w\-\s&']+ (founded|created) [A-Z][\w\-\s&']+\.?$",
    ]
    if any(re.match(pat, raw) for pat in templates):
        return True

    # Fallback: sentence has at least two named-entity-like chunks + relation verb.
    title_chunks = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", raw)
    has_relation = bool(
        re.search(r"\b(is|was|were|born|headquartered|based|founded|created)\b", lower)
    )
    return has_relation and len(title_chunks) >= 2


def memory_noise_ratio(memory: str) -> float:
    lines = [line.strip() for line in memory.splitlines() if line.strip()]
    if not lines:
        return 0.0
    fact_count = sum(1 for line in lines if strict_fact_signal(line))
    return max(0.0, min(1.0, 1.0 - (fact_count / len(lines))))


def explicit_noise_signal(text: str) -> bool:
    lower = (text or "").lower()
    if not lower.strip():
        return False
    cues = [
        "i am planning",
        "next month",
        "weekend",
        "bought",
        "show",
        "menu",
        "review",
        "tour",
        "delayed",
        "renovation",
    ]
    return any(cue in lower for cue in cues)


def postprocess_action(
    obs: LongHorizonMemoryObservation,
    action: LongHorizonMemoryAction,
    step: int,
    rewrite_count: int,
) -> LongHorizonMemoryAction:
    rel = relevance_score(obs.new_message)
    token_budget = int(obs.metadata.get("memory_token_budget", 160))
    strong_fact = strict_fact_signal(obs.new_message)
    noise_ratio = memory_noise_ratio(obs.memory)
    explicit_noise = explicit_noise_signal(obs.new_message)
    is_late_episode = step >= 8

    # Noop remains default; only override for explicit high-confidence fact relations.
    if action.operation == "noop" and strong_fact and obs.memory_count < int(0.90 * token_budget):
        return LongHorizonMemoryAction(operation="append")

    # In gather phase, avoid excessive noop for likely useful messages.
    if action.operation == "noop" and step <= 6 and rel >= 0.30 and not explicit_noise:
        return LongHorizonMemoryAction(operation="append")

    # Rewrite trigger: memory is either too large or too noisy.
    should_rewrite = (
        obs.memory_count >= int(0.50 * token_budget)
        or (obs.memory_count >= 26 and noise_ratio >= 0.35)
        or (step >= 7 and obs.memory_count >= 18 and noise_ratio >= 0.20)
    )

    # Late-stage compression opportunity: when memory is already sizable and
    # current message is noise, prefer one cleanup rewrite if none happened yet.
    late_cleanup_rewrite = (
        rewrite_count == 0
        and is_late_episode
        and explicit_noise
        and obs.memory_count >= 20
        and not strong_fact
    )
    should_rewrite = should_rewrite or late_cleanup_rewrite

    if action.operation != "rewrite" and should_rewrite:
        return LongHorizonMemoryAction(
            operation="rewrite",
            rewrite_memory=compress_memory_fact_preserving(obs.memory),
        )

    # Guardrail: prevent append spam; append only for high-confidence relevance.
    if action.operation == "append" and rel < 0.65:
        if strong_fact:
            return LongHorizonMemoryAction(operation="append")
        return LongHorizonMemoryAction(operation="noop")

    # Guardrail: if memory is near budget and model still appends, compress instead.
    if action.operation == "append" and obs.memory_count >= int(0.85 * token_budget):
        return LongHorizonMemoryAction(
            operation="rewrite",
            rewrite_memory=compress_memory(obs.memory),
        )

    # Guardrail: ensure rewrite payload is useful.
    if action.operation == "rewrite" and not (action.rewrite_memory or "").strip():
        return LongHorizonMemoryAction(
            operation="rewrite",
            rewrite_memory=compress_memory_fact_preserving(obs.memory),
        )

    return action


def parse_action(content: str, obs: LongHorizonMemoryObservation) -> LongHorizonMemoryAction:
    normalized = maybe_strip_code_fences(content)
    try:
        payload = json.loads(normalized)
        op = payload.get("operation", "noop")

        if op in {"append", "noop"}:
            return LongHorizonMemoryAction(operation=op)

        if op == "rewrite":
            rewrite_memory = payload.get("rewrite_memory")
            if isinstance(rewrite_memory, str):
                return LongHorizonMemoryAction(operation="rewrite", rewrite_memory=rewrite_memory)
            return LongHorizonMemoryAction(
                operation="rewrite",
                rewrite_memory=compress_memory(obs.memory),
            )
    except Exception:
        pass

    return heuristic_action(obs)


def ollama_chat(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout: float,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.1},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("content", "")


def choose_action(
    obs: LongHorizonMemoryObservation,
    base_url: str,
    model: str,
    timeout: float,
    step: int,
    rewrite_count: int,
) -> LongHorizonMemoryAction:
    user_prompt = (
        f"Task: {obs.task_name}\n"
        f"Current memory tokens: {obs.memory_count}\n"
        f"Current memory:\n{obs.memory}\n\n"
        f"New message:\n{obs.new_message}\n\n"
        "Decision rules:\n"
        "1) Gather useful facts early: do not overuse noop on potentially useful factual messages.\n"
        "2) Use noop only for clear distractors/noise.\n"
        "3) Use rewrite when memory is growing or noisy; preserve key facts needed for final QA.\n"
        "4) In late steps, if memory already has several facts and new message is noise, prefer one rewrite over noop.\n"
        "5) Do not force rewrite every episode; rewrite only when clearly beneficial.\n\n"
        "Return JSON action now."
    )

    try:
        content = ollama_chat(base_url=base_url, model=model, user_prompt=user_prompt, timeout=timeout)
        action = parse_action(content, obs)
        return postprocess_action(obs, action, step=step, rewrite_count=rewrite_count)
    except Exception:
        return postprocess_action(obs, heuristic_action(obs), step=step, rewrite_count=rewrite_count)


def check_ollama(base_url: str, model: str, timeout: float) -> None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        tags = response.json().get("models", [])
        names = {m.get("name", "") for m in tags}
        if model not in names:
            print(f"[WARN] Model '{model}' not found in Ollama tags. Available count={len(names)}")
        else:
            print(f"[OK] Found Ollama model: {model}")
    except Exception as exc:
        print(f"[WARN] Could not verify Ollama tags: {exc}")


def run_episode(
    env: LongHorizonMemoryEnvironment,
    base_url: str,
    model: str,
    timeout: float,
    max_steps: int,
) -> Dict[str, float]:
    obs = env.reset()
    total_reward = 0.0
    step_count = 0
    action_counts = {"append": 0, "rewrite": 0, "noop": 0}
    positive_append_rewards = 0
    total_appends = 0

    print("\n[EPISODE_START]")

    for step in range(1, max_steps + 1):
        step_count = step
        action = choose_action(
            obs,
            base_url=base_url,
            model=model,
            timeout=timeout,
            step=step,
            rewrite_count=action_counts["rewrite"],
        )
        obs = env.step(action)

        reward = float(obs.reward)
        score = float(obs.metadata.get("task_score", 0.0))
        breakdown = obs.metadata.get("reward_breakdown", {})

        total_reward += reward
        action_counts[action.operation] += 1
        if action.operation == "append":
            total_appends += 1
            if breakdown.get("append_relevance", 0.0) > 0:
                positive_append_rewards += 1
        print(
            f"[STEP] {step:02d} op={action.operation:<7} "
            f"reward={reward:+.3f} total={total_reward:+.3f} "
            f"score={score:.3f} done={obs.done}"
        )

        if obs.done:
            break

    final_score = float(obs.metadata.get("task_score", 0.0))
    append_positive_rate = positive_append_rewards / max(1, total_appends)
    print(
        f"[EPISODE_END] steps={step_count} total_reward={total_reward:+.3f} "
        f"final_score={final_score:.3f} actions={action_counts} append_positive_rate={append_positive_rate:.3f}"
    )

    return {
        "steps": float(step_count),
        "total_reward": total_reward,
        "final_score": final_score,
        "noop_rate": action_counts["noop"] / max(1, step_count),
        "rewrite_rate": action_counts["rewrite"] / max(1, step_count),
        "append_positive_rate": append_positive_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple Ollama inference against Long Horizon Memory env.")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    os.environ["LONG_HORIZON_MEMORY_TASK"] = args.task
    os.environ["LONG_HORIZON_MEMORY_SEED"] = str(args.seed)

    check_ollama(base_url=args.base_url, model=args.model, timeout=args.timeout)

    env = LongHorizonMemoryEnvironment()
    all_rewards = []
    all_scores = []
    all_noop_rates = []
    all_rewrite_rates = []
    all_append_positive_rates = []

    try:
        for i in range(args.episodes):
            os.environ["LONG_HORIZON_MEMORY_SEED"] = str(args.seed + i)
            result = run_episode(
                env=env,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_steps=args.max_steps,
            )
            all_rewards.append(result["total_reward"])
            all_scores.append(result["final_score"])
            all_noop_rates.append(result["noop_rate"])
            all_rewrite_rates.append(result["rewrite_rate"])
            all_append_positive_rates.append(result["append_positive_rate"])

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        avg_noop_rate = sum(all_noop_rates) / len(all_noop_rates) if all_noop_rates else 0.0
        avg_rewrite_rate = sum(all_rewrite_rates) / len(all_rewrite_rates) if all_rewrite_rates else 0.0
        avg_append_positive_rate = (
            sum(all_append_positive_rates) / len(all_append_positive_rates)
            if all_append_positive_rates
            else 0.0
        )
        print(
            f"\n[SUMMARY] episodes={len(all_rewards)} avg_total_reward={avg_reward:+.3f} "
            f"avg_final_score={avg_score:.3f} avg_noop_rate={avg_noop_rate:.3f} "
            f"avg_rewrite_rate={avg_rewrite_rate:.3f} "
            f"avg_append_positive_rate={avg_append_positive_rate:.3f}"
        )
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
