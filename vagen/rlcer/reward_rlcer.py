from __future__ import annotations

import json
import math
import re
import time
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from verl.utils.reward_score import default_compute_score
from vagen.rlcer.prompt_templates import NEW_SYSTEM_CONTENT, USER_CONTENT


def _strip_multimodal_placeholders(text: str) -> str:
    """Remove common multimodal placeholders from decoded chat text.

    When building auxiliary rubricator prompts from decoded token ids, residual
    vision placeholders (e.g. Qwen-VL image tokens) can be preserved in text.
    Sending such text without matching `image_data` may trigger SGLang/processor
    index errors. This helper strips those placeholders for text-only prompts.
    """
    if not text:
        return ""

    s = str(text)
    for token in (
        "<image>",
        "[image]",
        "<img>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|audio_pad|>",
    ):
        s = s.replace(token, " ")

    s = re.sub(r"<\|[^>]*?(image|vision|video|audio)[^>]*\|>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NEGATIVE_CUE_WORDS = {
    "fail",
    "fails",
    "failed",
    "error",
    "errors",
    "incorrect",
    "wrong",
    "missing",
    "redundant",
    "irrelevant",
    "hallucinate",
    "hallucination",
}


def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _format_history_items(items: List[dict], max_items: int = 4, max_chars: int = 1600) -> str:
    if len(items) == 0:
        return "None (Initial Turn)"

    trimmed = items[-max_items:]
    blocks: List[str] = []
    for it in trimmed:
        tid = it.get("turn_idx")
        structured = _extract_structured_sections(str(it.get("solution", "")))
        blocks.append(
            "\n".join(
                [
                    f"[Turn {tid}]",
                    "observation:",
                    structured["observation"],
                    "think:",
                    structured["think"],
                    "answer:",
                    structured["answer"],
                    "prediction:",
                    structured["prediction"],
                ]
            )
        )

    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _extract_structured_sections(solution: str) -> Dict[str, str]:
    """Parse solver response into observation/think/answer/prediction sections.

    Strict section names are preferred. Falls back to robust heuristics if sections
    are partially missing.
    """
    s = (solution or "").strip()
    if not s:
        return {"observation": "", "think": "", "answer": "", "prediction": ""}

    # Normalize full-width colon for robustness.
    s = s.replace("：", ":")
    lower = s.lower()

    # Candidate heading aliases mapped to canonical key.
    heading_aliases = {
        "observation": ["observation", "观察", "obs"],
        "think": ["think", "reasoning", "thought", "思考", "analysis"],
        "answer": ["answer", "action", "动作", "结论"],
        "prediction": ["prediction", "predict", "预测"],
    }

    # Locate first occurrence of each canonical heading.
    positions: List[tuple[int, str, int]] = []
    for key, aliases in heading_aliases.items():
        best_pos = None
        best_len = 0
        for a in aliases:
            m = re.search(rf"(?im)(^|\n)\s*(?:\[?\s*){re.escape(a)}\s*\]?\s*:\s*", lower)
            if m is not None:
                p = m.start()
                if best_pos is None or p < best_pos:
                    best_pos = p
                    best_len = len(m.group(0))
        if best_pos is not None:
            positions.append((best_pos, key, best_len))

    positions.sort(key=lambda x: x[0])
    out = {"observation": "", "think": "", "answer": "", "prediction": ""}

    if positions:
        for i, (start, key, hlen) in enumerate(positions):
            body_start = start + hlen
            body_end = positions[i + 1][0] if i + 1 < len(positions) else len(s)
            content = s[body_start:body_end].strip()
            out[key] = content

    # Fallbacks if headings are missing
    if not out["answer"]:
        _, ans = _extract_cot_and_answer(s)
        out["answer"] = ans
    if not out["think"]:
        out["think"] = s[:500]
    if not out["observation"]:
        out["observation"] = "(not explicitly provided)"
    if not out["prediction"]:
        out["prediction"] = "(not explicitly provided)"

    # Keep sections concise for prompt budget
    for k in out.keys():
        if len(out[k]) > 600:
            out[k] = out[k][:600] + "..."
    return out


def _extract_concat_turn_candidates(text: str, max_turns: int = 6) -> List[Dict[str, str]]:
    """Best-effort extraction of multiple structured turns from concat-mode text."""
    s = (text or "").replace("：", ":")
    if not s.strip():
        return []

    # Split by repeated observation heading as turn boundary.
    chunks = re.split(r"(?im)(?=^\s*\[?\s*observation\s*\]?\s*:)", s)
    turns: List[Dict[str, str]] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        sec = _extract_structured_sections(c)
        # require at least one non-fallback signal
        if sec["answer"] or sec["think"]:
            turns.append(sec)
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    return turns


def _format_structured_history_from_sections(sections: List[Dict[str, str]], max_chars: int = 1800) -> str:
    if not sections:
        return "None (Initial Turn)"
    blocks: List[str] = []
    for i, sec in enumerate(sections, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Turn {i}]",
                    "observation:",
                    sec.get("observation", ""),
                    "think:",
                    sec.get("think", ""),
                    "answer:",
                    sec.get("answer", ""),
                    "prediction:",
                    sec.get("prediction", ""),
                ]
            )
        )
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _build_trajectory_history_for_items(items: List[dict]) -> List[str]:
    """Build per-sample trajectory history from batch trajectories.

    Priority:
    1) Same (group, traj_idx) and previous turn_idx samples in current batch
    2) Fallback to decoded prompt_str from training sample
    """
    group_traj_to_indices: Dict[tuple[str, Any], List[int]] = {}
    for i, x in enumerate(items):
        key = (x["group"], x.get("traj_idx"))
        group_traj_to_indices.setdefault(key, []).append(i)

    histories = ["None (Initial Turn)"] * len(items)

    # Detect mode: if turn_idx is mostly available, treat as no-concat.
    has_turn = sum(1 for x in items if x.get("turn_idx") is not None)
    no_concat_mode = has_turn >= max(1, len(items) // 2)

    if not no_concat_mode:
        # Concat-mode optimization: derive history from serialized text of each sample.
        for i, x in enumerate(items):
            combined_text = f"{x.get('prompt_str', '')}\n{x.get('solution', '')}"
            turns = _extract_concat_turn_candidates(combined_text)
            # For current rubric generation, history should be previous turns only.
            if len(turns) >= 2:
                histories[i] = _format_structured_history_from_sections(turns[:-1])
            elif len(turns) == 1:
                histories[i] = "None (Initial Turn)"
            else:
                prompt_str = str(x.get("prompt_str", "")).strip()
                histories[i] = prompt_str if prompt_str else "None (Initial Turn)"
        return histories

    # no-concat mode: use prior turns from same (group, traj_idx).
    for _, idxs in group_traj_to_indices.items():
        idxs_sorted = sorted(
            idxs,
            key=lambda t: (
                items[t].get("turn_idx") if items[t].get("turn_idx") is not None else 10**9,
                t,
            ),
        )

        prev_items: List[dict] = []
        for t in idxs_sorted:
            cur_turn = items[t].get("turn_idx")
            if cur_turn is None:
                # Unknown turn index: fallback to prompt text directly
                prompt_str = str(items[t].get("prompt_str", "")).strip()
                histories[t] = prompt_str if prompt_str else "None (Initial Turn)"
                continue

            hist = _format_history_items(prev_items)
            histories[t] = hist
            prev_items.append(items[t])

    # final fallback for any empty history
    for i in range(len(items)):
        if not histories[i] or histories[i] == "None (Initial Turn)":
            prompt_str = str(items[i].get("prompt_str", "")).strip()
            if prompt_str:
                histories[i] = prompt_str
    return histories


def _compose_rubricator_system_content(trajectory_history: str) -> str:
    history_text = trajectory_history.strip() if trajectory_history else "None (Initial Turn)"
    marker = "## [Trajectory History] None (Initial Turn)"
    if marker in NEW_SYSTEM_CONTENT:
        return NEW_SYSTEM_CONTENT.replace(marker, f"## [Trajectory History] {history_text}")
    # robust fallback
    return NEW_SYSTEM_CONTENT + f"\n\n## [Trajectory History] {history_text}"


# ---------------------------------------------------------------------------
# Rubric weight constraints (feature: normalize_rubric_weights)
# ---------------------------------------------------------------------------
# Clamp individual rubric weights to prevent inflation / extreme values.
_MAX_RUBRIC_WEIGHT = 20.0
_MIN_RUBRIC_WEIGHT = -20.0
# After clamping, positive weights are L1-normalized so sum(positive) = 1.0
# This prevents the model from gaming cot_reward by assigning huge weights.
_NORMALIZE_POSITIVE_WEIGHTS = True


def _clamp_and_normalize_rubrics(rubrics: List[Rubric], max_abs_weight: float = _MAX_RUBRIC_WEIGHT) -> List[Rubric]:
    """Clamp rubric weights and L1-normalize positive weights.

    1) Clamp each weight to [-max_abs_weight, max_abs_weight]
    2) If _NORMALIZE_POSITIVE_WEIGHTS, scale positive weights so they sum to 1.0
       Negative weights are kept as-is (they represent penalties).
       This ensures cot_reward is in a predictable range regardless of the
       absolute weight values the model generates.
    """
    clamped = []
    for r in rubrics:
        pts = max(-max_abs_weight, min(max_abs_weight, r.points))
        clamped.append(Rubric(criterion=r.criterion, points=pts, rule_id=r.rule_id))

    if not _NORMALIZE_POSITIVE_WEIGHTS:
        return clamped

    pos_sum = sum(r.points for r in clamped if r.points > 0)
    if pos_sum <= 0:
        return clamped

    normalized = []
    for r in clamped:
        if r.points > 0:
            normalized.append(Rubric(criterion=r.criterion, points=r.points / pos_sum, rule_id=r.rule_id))
        else:
            normalized.append(r)
    return normalized


@dataclass
class Rubric:
    criterion: str
    points: float
    rule_id: str = ""


@dataclass
class RubricProposal:
    rubrics: List[Rubric]
    format_ok: bool
    raw: str = ""


_RULE_ID_ALIASES = {
    "ACTION_VALIDITY": "ACTION_LEGALITY",
    "LEGAL_ACTION": "ACTION_LEGALITY",
    "LEGALITY": "ACTION_LEGALITY",
    "SPATIAL": "SPATIAL_GROUNDING",
    "GOAL_PROGRESS": "STRATEGIC_PROGRESS",
}


def canonicalize_rule_id(rule_id: Any) -> str:
    """Map model-generated rule IDs onto stable rule-family identifiers."""
    rid = str(rule_id or "").strip().upper()
    rid = re.sub(r"[^A-Z0-9]+", "_", rid).strip("_")
    return _RULE_ID_ALIASES.get(rid, rid)


def _rule_corr_key(data_source: Any, rule_id: Any, schema_version: str = "v1") -> str:
    """Namespace correlation statistics by task and rubric schema."""
    source = re.sub(r"[^a-z0-9]+", "_", str(data_source or "default").strip().lower()).strip("_")
    return f"{source or 'default'}|{canonicalize_rule_id(rule_id)}|{schema_version}"


class BaseRubricator:
    def generate(self, question: str, response: str, cot: str, **kwargs) -> RubricProposal:
        raise NotImplementedError


class BaseVerifier:
    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        raise NotImplementedError


def _safe_json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^[^`]*```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```[^`]*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Try to repair truncated JSON by closing open brackets
    for closing in ["]}", "}", "]"]:
        try:
            return json.loads(candidate + closing)
        except Exception:
            continue

    # Fallback: try ast.literal_eval for Python dict repr (single quotes, True/False/None)
    try:
        import ast
        result = ast.literal_eval(candidate)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return None


def _extract_rubrics_from_malformed_text(raw: str, max_rubrics: int) -> List[Rubric]:
    """Fallback extractor using regex to pull individual rubric entries from
    malformed text where JSON / ast parsing fails.

    Handles:
    - Python repr with single quotes and escaped quotes (\\')
    - Corrupted first rubric but valid subsequent rubrics
    - Mixed quote styles
    """
    # Remove backslash escape artifacts from Python repr
    cleaned = raw.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")

    rubrics: List[Rubric] = []

    # Split by rule_id markers to isolate each rubric segment
    segments = re.split(r"""['"]rule_id['"]\s*:\s*['"]""", cleaned)

    for seg in segments[1:]:  # skip text before first rule_id
        # Extract rule_id (everything up to the closing quote)
        rule_id_match = re.match(r"""([^'"]+)['"]""", seg)
        if not rule_id_match:
            continue
        rule_id = canonicalize_rule_id(rule_id_match.group(1))

        # Extract weight — the most reliable anchor (always a number)
        weight_match = re.search(r"""['"]weight['"]\s*:\s*([+-]?\d+\.?\d*)""", seg)
        if not weight_match:
            continue
        weight = float(weight_match.group(1))

        # Extract description: text between 'description': and 'weight':
        desc_start_match = re.search(r"""['"]description['"]\s*:\s*['"]""", seg)
        desc_text = ""
        if desc_start_match:
            desc_start = desc_start_match.end()
            desc_end = weight_match.start()
            desc_text = seg[desc_start:desc_end].strip()
            # Strip trailing quote + optional comma/whitespace
            desc_text = re.sub(r"""['"][,]?\s*$""", "", desc_text).strip()

        if not desc_text:
            continue

        rubrics.append(Rubric(criterion=desc_text, points=weight, rule_id=rule_id))

    return rubrics[:max_rubrics]


def _parse_rubric_proposal_from_raw(raw: str, max_rubrics: int) -> RubricProposal:
    obj = _safe_json_extract(raw)

    if not isinstance(obj, dict):
        # JSON / ast parsing failed — try regex fallback extraction
        rubrics = _extract_rubrics_from_malformed_text(raw, max_rubrics)
        if rubrics:
            
            return RubricProposal(rubrics=rubrics, format_ok=True, raw=raw)
        
        return RubricProposal(rubrics=[], format_ok=False, raw=raw)

    rubrics_raw = obj.get("rubrics", [])
    # if not rubrics_raw:
    #     print(f"[RLCER] rubric JSON has no 'rubrics' field, keys: {list(obj.keys())}, raw preview: {raw[:200]!r}")
    rubrics: List[Rubric] = []
    for item in rubrics_raw:
        if not isinstance(item, dict):
            continue
        # Support both legacy and new schema.
        crit = str(item.get("criterion", item.get("description", ""))).strip()
        pts = item.get("points", item.get("weight", 0))
        rule_id = canonicalize_rule_id(item.get("rule_id", ""))
        if not crit:
            continue
        try:
            pts = float(pts)
        except Exception:
            continue
        if pts == 0:
            continue
        rubrics.append(Rubric(criterion=crit, points=pts, rule_id=rule_id))

    # If parsed JSON had rubrics field but all entries were invalid, try regex
    if not rubrics:
        rubrics = _extract_rubrics_from_malformed_text(raw, max_rubrics)
        # if rubrics:
        #     print(f"[RLCER] regex fallback extracted {len(rubrics)} rubrics (JSON rubrics field was empty/invalid)")
   
    rubrics = rubrics[: max_rubrics]
    rubrics = _clamp_and_normalize_rubrics(rubrics)
    return RubricProposal(rubrics=rubrics, format_ok=len(rubrics) > 0, raw=raw)


def _extract_policy_rubric_proposal_from_extra(extra_info: dict, max_rubrics: int) -> Optional[RubricProposal]:
    if not isinstance(extra_info, dict):
        return None
    raw = extra_info.get("rlcer_policy_rubric_raw", None)
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    raw = str(raw)

    if not raw.strip():
        return RubricProposal(rubrics=[], format_ok=False, raw=raw)
    return _parse_rubric_proposal_from_raw(raw=raw, max_rubrics=max_rubrics)


def _extract_cot_and_answer(solution: str) -> tuple[str, str]:
    if solution is None:
        return "", ""
    s = str(solution)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if boxed:
        return s, boxed[-1].strip()

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "", ""
    answer = lines[-1]
    return s, answer


def _normalize_token_set(text: str) -> set[str]:
    text = (text or "").lower()
    toks = re.findall(r"[a-zA-Z0-9_]{3,}", text)
    return set(toks)


class HeuristicRubricator(BaseRubricator):
    """Simple fallback rubricator for bootstrapping/debugging.

    Produces parseable rubrics without external dependencies.
    """

    def __init__(self, max_rubrics: int = 6):
        self.max_rubrics = max(1, int(max_rubrics))

    def generate(self, question: str, response: str, cot: str, **kwargs) -> RubricProposal:
        rubrics = [
            Rubric("Reasoning steps are coherent and logically connected", 3, rule_id="COHERENCE"),
            Rubric("Solution includes at least one explicit verification/check", 2, rule_id="VERIFICATION"),
            Rubric("Reasoning stays focused on the target question", 2, rule_id="FOCUS"),
            Rubric("Reasoning contains arithmetic/algebraic inconsistency", -3, rule_id="INCONSISTENCY"),
            Rubric("Reasoning is repetitive without adding new progress", -2, rule_id="REPETITION"),
            Rubric("Final conclusion is clearly stated", 2, rule_id="CONCLUSION"),
        ][: self.max_rubrics]
        return RubricProposal(rubrics=rubrics, format_ok=True, raw="heuristic")


class HeuristicVerifier(BaseVerifier):
    """Heuristic verifier with deterministic lexical checks.

    Handles both short descriptions and structured rubric descriptions from
    ray_trainer.py's policy rubricator (which include rule_id prefixes like
    SPATIAL_GROUNDING, ACTION_LEGALITY, etc.).

    For structured rubrics, we extract checkable keywords and verify they
    appear in the solver's response. This is intentionally simple and should
    be replaced by an external verifier model for high-fidelity judgement.
    """

    # rule_id → relevant response section keywords for verification
    _RULE_SECTIONS = {
        "SPATIAL_GROUNDING": ["observation", "obs"],
        "ACTION_LEGALITY": ["answer", "action", "动作"],
        "LOGIC_FRESHNESS": ["think", "reasoning", "思考"],
        "STRATEGIC_PROGRESS": ["think", "reasoning", "思考", "answer", "action", "动作"],
        "FATAL_CORNER": ["answer", "action", "动作"],
    }

    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        # Also accept rubric objects (with rule_id) passed via kwargs
        rubrics = kwargs.get("rubrics", None)

        # Build structured sections from response
        structured = _extract_structured_sections(response or "")

        out: List[bool] = []
        cot_tokens = _normalize_token_set(cot)

        for idx, c in enumerate(criteria):
            crit = c or ""

            # Try to determine the rule_id for this criterion
            rule_id = ""
            if rubrics and idx < len(rubrics):
                rule_id = getattr(rubrics[idx], "rule_id", "") if hasattr(rubrics[idx], "rule_id") else ""

            if rule_id and rule_id in self._RULE_SECTIONS:
                # Structured rubric from policy rubricator: check relevant
                # response section against criterion keywords.
                section_keys = self._RULE_SECTIONS[rule_id]
                section_text = " ".join(
                    structured.get(k, "") for k in section_keys if structured.get(k)
                )
                section_tokens = _normalize_token_set(section_text)
                crit_tokens = _normalize_token_set(crit)

                # For negative criteria, check if the flaw keywords appear
                crit_lower = crit.lower()
                is_negative = any(w in crit_lower for w in NEGATIVE_CUE_WORDS)

                if is_negative:
                    # Flaw present → penalty rubric is satisfied
                    overlap = len(section_tokens & crit_tokens)
                    out.append(overlap >= 2)
                else:
                    # For positive criteria: PASS if the relevant section
                    # contains keywords that match the rubric's focus.
                    # Extract key nouns/verbs from criterion (skip weight/point markers)
                    clean_crit = re.sub(r"\(?\d+\s*(?:pts?|points?)\)?", "", crit)
                    clean_crit = re.sub(r"\bPASS\b|\bFAIL\b|\bauto-PASS\b|\bScore 0\b", "", clean_crit)
                    clean_tokens = _normalize_token_set(clean_crit)
                    overlap = len(section_tokens & clean_tokens)
                    out.append(overlap >= 1)
            else:
                # Original heuristic: token overlap between criterion and CoT
                crit_tokens = _normalize_token_set(crit)
                overlap = len(cot_tokens & crit_tokens)
                thresh = 1 if len(crit_tokens) < 8 else 2
                base_sat = overlap >= thresh

                crit_lower = crit.lower()
                is_negative = any(w in crit_lower for w in NEGATIVE_CUE_WORDS)
                if is_negative:
                    response_lower = (response or "").lower()
                    flaw_detected = any(w in response_lower for w in NEGATIVE_CUE_WORDS)
                    out.append(flaw_detected)
                else:
                    out.append(bool(base_sat))
        return out


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: float = 60.0):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        # A verifier at 127.0.0.1 is an in-node service, never an Internet
        # request.  Inheriting cluster HTTP(S)_PROXY variables can route its raw
        # traffic through a proxy and make Uvicorn report "Invalid HTTP request".
        self._client = httpx.Client(timeout=timeout, trust_env=False)
        self._request_error_count = 0

    def chat(self, messages: List[dict], temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0) -> str:
        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        last_exc = None
        for attempt in range(3):
            try:
                r = self._client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                obj = r.json()
                return obj["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                last_exc = e
            except (httpx.RequestError, ConnectionError, OSError, IOError) as e:
                last_exc = e
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        self._request_error_count += 1
        if self._request_error_count <= 3:
            print(
                f"[external-verifier] request failed after 3 attempts: "
                f"{type(last_exc).__name__}: {last_exc}",
                flush=True,
            )
        raise last_exc


def _pil_to_data_url(image: Any, image_format: str = "PNG") -> Optional[str]:
    try:
        from PIL import Image

        if not isinstance(image, Image.Image):
            return None
        buf = io.BytesIO()
        image.save(buf, format=image_format)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _extract_first_image_from_kwargs(kwargs: dict) -> Any:
    extra = kwargs.get("extra_info") or {}
    # priority: explicit field
    if "image" in extra:
        return extra.get("image")
    # common vagen field from agent-loop extra_fields
    if "image_data" in extra:
        img_data = extra.get("image_data")
        if isinstance(img_data, list) and len(img_data) > 0:
            # For a multi-turn trajectory the last collected image is the
            # observation used for the latest solver decision. Using index 0
            # would make an external verifier judge later-turn claims against
            # the initial board.
            return img_data[-1]
        return img_data
    return None


class LLMRubricator(BaseRubricator):
    def __init__(self, cfg: dict):
        self.client = OpenAICompatibleClient(
            base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            model=cfg.get("model", ""),
            api_key=cfg.get("api_key"),
            timeout=float(cfg.get("timeout", 60.0)),
        )
        self.max_rubrics = int(cfg.get("max_rubrics", 8))
        self.temperature = float(cfg.get("temperature", 0.3))
        self.top_p = float(cfg.get("top_p", 0.9))

    def generate(self, question: str, response: str, cot: str, **kwargs) -> RubricProposal:
        image = _extract_first_image_from_kwargs(kwargs)
        image_url = _pil_to_data_url(image)

        trajectory_history = str(kwargs.get("trajectory_history", "") or "None (Initial Turn)")
        turn_idx = kwargs.get("turn_idx", None)
        if turn_idx is None:
            turn_id = "turn_000001"
        else:
            turn_id = f"turn_{int(turn_idx):06d}" if _to_int_or_none(turn_idx) is not None else "turn_000001"

        # Strip multimodal placeholders from response text so that vision tokens
        # (e.g. Qwen-VL <image>/<|image_pad|>) don't leak into the rubricator
        # prompt when the response is included as text.
        stripped_response = _strip_multimodal_placeholders(response or "")

        # Build user_tail matching the structure used by
        # ray_trainer._build_rlcer_policy_rubricator_prompts.
        user_tail = (
            f"\n\n[Turn ID]\n{turn_id}\n"
            f"\n[Solver Response]\n{stripped_response}\n"
            "\n[Constraint]\nReturn ONLY the JSON in a json block."
        )

        system_msg = {
            "role": "system",
            "content": _compose_rubricator_system_content(trajectory_history),
        }

        if image_url is not None:
            # Match ray_trainer.py structure: [Initial Observation] + image + prompt + user_tail
            user_content = [
                {"type": "text", "text": "[Initial Observation]:\n"},
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": "\nAnalyze the current Sokoban board state and generate evaluation rubrics." + user_tail,
                },
            ]
            user_msg = {"role": "user", "content": user_content}
        else:
            # Text-only fallback: use USER_CONTENT template + user_tail
            user_msg = {
                "role": "user",
                "content": f"{USER_CONTENT}\n{user_tail}",
            }

        raw = self.client.chat([system_msg, user_msg], temperature=self.temperature, top_p=self.top_p, max_tokens=1500)
        return _parse_rubric_proposal_from_raw(raw=raw, max_rubrics=self.max_rubrics)


class LLMVerifier(BaseVerifier):
    def __init__(self, cfg: dict):
        self.single_criterion_requests = str(
            os.getenv("VERIFIER_SINGLE_CRITERION_REQUESTS", cfg.get("single_criterion_requests", "false"))
        ).lower() in {"1", "true", "yes", "on"}
        self.group_concurrency = max(
            1,
            int(os.getenv("VERIFIER_GROUP_CONCURRENCY", cfg.get("group_concurrency", 1))),
        )
        self.request_concurrency = max(
            1,
            int(os.getenv("VERIFIER_REQUEST_CONCURRENCY", cfg.get("request_concurrency", 1))),
        )
        self.max_tokens = max(
            16,
            int(os.getenv("VERIFIER_MAX_TOKENS", cfg.get("max_tokens", 1000))),
        )
        self.client = OpenAICompatibleClient(
            # Environment overrides let thin task launchers switch an existing
            # grounded-verifier training script to an external endpoint without
            # editing or duplicating the original long Hydra command.
            base_url=os.getenv("VERIFIER_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8000/v1")),
            model=os.getenv("VERIFIER_MODEL", cfg.get("model", "")),
            api_key=os.getenv("VERIFIER_API_KEY", cfg.get("api_key")),
            timeout=float(os.getenv("VERIFIER_TIMEOUT", cfg.get("timeout", 60.0))),
        )

    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        if self.single_criterion_requests and len(criteria) > 1:
            rubrics = kwargs.get("rubrics") or []
            results = []
            for idx, criterion in enumerate(criteria):
                one_kwargs = dict(kwargs)
                one_kwargs["rubrics"] = [rubrics[idx]] if idx < len(rubrics) else []
                results.extend(self._judge_once([criterion], question, response, cot, **one_kwargs))
            return results
        return self._judge_once(criteria, question, response, cot, **kwargs)

    def _judge_once(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        image = _extract_first_image_from_kwargs(kwargs)
        image_url = _pil_to_data_url(image)
        rubrics: Optional[List[Rubric]] = kwargs.get("rubrics", None)

        stripped_response = _strip_multimodal_placeholders(response or "")

        annotated = []
        for idx, c in enumerate(criteria):
            if rubrics and idx < len(rubrics) and hasattr(rubrics[idx], "points"):
                prefix = "[+] " if rubrics[idx].points > 0 else "[-] "
            else:
                prefix = ""
            annotated.append(prefix + c)

        text_content = (
            "Evaluate each criterion as strict True/False against the response reasoning.\n"
            "[+] criteria: True = quality IS present. [-] criteria: True = flaw IS present.\n"
            "You MUST use the provided image to verify factual claims about object positions, "
            "actions, and spatial relationships described in the response.\n"
            "Return JSON only: {\"judgement\": [bool, ...]} with same order.\n"
            f"Question:\n{question}\n\n"
            f"Response:\n{stripped_response}\n\n"
            f"Criteria:\n{json.dumps(annotated, ensure_ascii=False)}"
        )

        if image_url is not None:
            user_content = [
                {"type": "text", "text": "[Visual Context]:\n"},
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "\n" + text_content},
            ]
        else:
            user_content = text_content

        prompt = {"role": "user", "content": user_content}
        raw = self.client.chat([prompt], temperature=0.0, max_tokens=self.max_tokens)
        obj = _safe_json_extract(raw)
        if not isinstance(obj, dict):
            return [False] * len(criteria)
        vals = obj.get("judgement", [])
        if not isinstance(vals, list):
            return [False] * len(criteria)
        out = [bool(x) for x in vals[: len(criteria)]]
        if len(out) < len(criteria):
            out.extend([False] * (len(criteria) - len(out)))

        return out


_RUBRICATOR_CACHE: dict[str, BaseRubricator] = {}
_VERIFIER_CACHE: dict[str, BaseVerifier] = {}


def _make_cache_key(mode: str, cfg: dict) -> str:
    # OmegaConf DictConfig is not JSON-serializable; convert to plain dict first.
    cfg_plain = dict(cfg) if not isinstance(cfg, dict) else cfg
    return json.dumps({"mode": mode, "cfg": cfg_plain}, sort_keys=True, ensure_ascii=False)


def _build_rubricator(rubricator_cfg: dict) -> BaseRubricator:
    mode = str(rubricator_cfg.get("mode", "policy")).lower()
    key = _make_cache_key(mode, rubricator_cfg)
    if key in _RUBRICATOR_CACHE:
        return _RUBRICATOR_CACHE[key]

    if mode in {"heuristic", "rule"}:
        inst: BaseRubricator = HeuristicRubricator(max_rubrics=int(rubricator_cfg.get("max_rubrics", 6)))
    else:
        # policy/external both use an OpenAI-compatible endpoint; in policy mode,
        # point this endpoint to the same model service as the actor policy.
        inst = LLMRubricator(rubricator_cfg)

    _RUBRICATOR_CACHE[key] = inst
    return inst


class GroundedVerifier(BaseVerifier):
    """Verifies rubric criteria using environment ground truth (100% accuracy).

    Verification logic is fixed per rule_id — does not parse criteria text.
    LLM Rubricator only controls weight allocation across dimensions.
    """

    VERTICAL_MAP = {"above": -1, "below": 1, "same row": 0}
    HORIZONTAL_MAP = {"left": -1, "right": 1, "same column": 0}

    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        extra_info = kwargs.get("extra_info", {})
        turn_env_states = extra_info.get("turn_env_states", [])
        initial_env_state = extra_info.get("initial_env_state", None)
        verifier_context = extra_info.get("verifier_context", {}) or {}
        rubrics = kwargs.get("rubrics", [])

        # SVG exposes independent raster/structure features through
        # verifier_context rather than turn_env_states.  Dispatch explicitly,
        # while accepting old SVG rollout records that predate task_type.
        if isinstance(verifier_context, dict) and (
            verifier_context.get("task_type") == "svg"
            or "valid_svg" in verifier_context
        ):
            return SVGGroundedVerifier().judge_many(
                criteria=criteria,
                question=question,
                response=response,
                cot=cot,
                rubrics=rubrics,
                extra_info=extra_info,
            )

        if not turn_env_states and not initial_env_state:
            return [False] * len(criteria)

        results = []
        is_frozenlake = bool(
            (initial_env_state or {}).get("task_type") == "frozenlake"
            or any(s.get("task_type") == "frozenlake" for s in turn_env_states)
        )
        for idx in range(len(criteria)):
            rubric = rubrics[idx] if idx < len(rubrics) else None
            rule_id = getattr(rubric, "rule_id", "") if rubric else ""
            if is_frozenlake:
                satisfied = self._judge_frozenlake_by_rule_id(
                    rule_id,
                    response,
                    turn_env_states,
                    initial_env_state,
                )
            else:
                # Preserve the existing Sokoban behavior for unmarked states.
                satisfied = self._judge_by_rule_id(
                    rule_id,
                    response,
                    turn_env_states,
                    initial_env_state,
                )
            results.append(satisfied)
        return results

    def _judge_frozenlake_by_rule_id(
        self,
        rule_id: str,
        response: str,
        turn_env_states: List[dict],
        initial_env_state: Optional[dict],
    ) -> bool:
        if rule_id == "SPATIAL_GROUNDING":
            return self._check_frozenlake_spatial(
                response, turn_env_states, initial_env_state
            )
        if rule_id == "ACTION_LEGALITY":
            return self._check_legality(turn_env_states)
        if rule_id in ("LOGIC_FRESHNESS", "PREDICTION_ACCURACY"):
            return self._check_frozenlake_prediction(response, turn_env_states)
        if rule_id == "STRATEGIC_PROGRESS":
            return self._check_frozenlake_progress(
                turn_env_states, initial_env_state
            )
        if rule_id == "FATAL_CORNER":
            return bool(turn_env_states) and not any(
                state.get("fell_in_hole", False) for state in turn_env_states
            )
        return any(s.get("action_is_effective", False) for s in turn_env_states)

    def _check_frozenlake_spatial(
        self,
        response: str,
        turn_env_states: List[dict],
        initial_env_state: Optional[dict],
    ) -> bool:
        obs_match = re.search(r"<observation>(.*?)</observation>", response, re.DOTALL)
        if not obs_match:
            return False
        obs_text = obs_match.group(1).lower()
        ref_state = initial_env_state or (
            turn_env_states[0] if turn_env_states else None
        )
        if not ref_state or not ref_state.get("player_pos"):
            return False

        goal_desc = self._parse_relative_position(obs_text, "goal")
        goals = ref_state.get("goal_positions", [])
        if not goal_desc or not any(
            self._verify_position(goal_desc, ref_state["player_pos"], goal)
            for goal in goals
        ):
            return False

        # If the response makes a hole-position claim, it must match at least
        # one actual hole.  Goal grounding remains mandatory in every state.
        if "hole" in obs_text:
            hole_desc = self._parse_relative_position(obs_text, "hole")
            holes = ref_state.get("hole_positions", [])
            if not hole_desc or not any(
                self._verify_position(hole_desc, ref_state["player_pos"], hole)
                for hole in holes
            ):
                return False
        return True

    def _check_frozenlake_prediction(
        self,
        response: str,
        turn_env_states: List[dict],
    ) -> bool:
        pred_match = re.search(r"<prediction>(.*?)</prediction>", response, re.DOTALL)
        if not pred_match or not turn_env_states:
            return False
        pred_text = pred_match.group(1).lower()
        end_state = turn_env_states[-1]
        if end_state.get("reached_goal", False):
            return "goal" in pred_text and any(
                cue in pred_text for cue in ("reach", "reached", "on goal", "victory")
            )
        if end_state.get("fell_in_hole", False):
            return "hole" in pred_text and any(
                cue in pred_text for cue in ("fall", "fell", "in hole", "on hole")
            )

        goal_desc = self._parse_relative_position(pred_text, "goal")
        return bool(
            goal_desc
            and end_state.get("player_pos")
            and any(
                self._verify_position(
                    goal_desc, end_state["player_pos"], goal
                )
                for goal in end_state.get("goal_positions", [])
            )
        )

    @staticmethod
    def _check_frozenlake_progress(
        turn_env_states: List[dict],
        initial_env_state: Optional[dict],
    ) -> bool:
        if not turn_env_states:
            return False
        end_state = turn_env_states[-1]
        if end_state.get("fell_in_hole", False):
            return False
        if end_state.get("reached_goal", False):
            return True
        start_state = initial_env_state or turn_env_states[0]
        start_distance = start_state.get("goal_distance")
        end_distance = end_state.get("goal_distance")
        if start_distance is None or end_distance is None:
            return False
        return float(end_distance) < float(start_distance)

    def _judge_by_rule_id(self, rule_id: str, response: str, turn_env_states: List[dict], initial_env_state: Optional[dict]) -> bool:
        if rule_id == "SPATIAL_GROUNDING":
            return self._check_spatial(response, turn_env_states, initial_env_state)
        elif rule_id == "ACTION_LEGALITY":
            return self._check_legality(turn_env_states)
        elif rule_id in ("LOGIC_FRESHNESS", "PREDICTION_ACCURACY"):
            return self._check_prediction(response, turn_env_states)
        elif rule_id == "STRATEGIC_PROGRESS":
            return self._check_progress(turn_env_states, initial_env_state)
        elif rule_id in ("PUSH_SIDE_POSITIONING", "PRE_TARGET_ALIGNMENT"):
            return self._check_progress(turn_env_states, initial_env_state)
        # Unknown rule families must not inherit ACTION_EFFECTIVE semantics.
        # In particular, negative heuristic rubrics such as INCONSISTENCY and
        # REPETITION would otherwise penalize successful Sokoban actions.
        return False

    def _check_spatial(self, response: str, turn_env_states: List[dict], initial_env_state: Optional[dict]) -> bool:
        """Verify <observation> text matches actual object positions."""
        obs_match = re.search(r"<observation>(.*?)</observation>", response, re.DOTALL)
        if not obs_match:
            return False
        obs_text = obs_match.group(1).lower()

        ref_state = initial_env_state if initial_env_state else (turn_env_states[0] if turn_env_states else None)
        if not ref_state:
            return False

        player_pos = ref_state.get("player_pos")
        box_positions = ref_state.get("box_positions", [])
        target_positions = ref_state.get("target_positions", [])
        if not player_pos:
            return False

        correct_count = 0
        total_count = 0

        for box_pos in box_positions:
            total_count += 1
            described = self._parse_relative_position(obs_text, "box")
            if described and self._verify_position(described, player_pos, box_pos):
                correct_count += 1

        for tgt_pos in target_positions:
            total_count += 1
            described = self._parse_relative_position(obs_text, "target")
            if described and self._verify_position(described, player_pos, tgt_pos):
                correct_count += 1

        if total_count == 0:
            return False
        return correct_count / total_count >= 0.5

    def _check_legality(self, turn_env_states: List[dict]) -> bool:
        """Check if all actions in the trajectory were valid and effective."""
        if not turn_env_states:
            return False
        valid_count = sum(1 for s in turn_env_states if s.get("action_is_valid", False))
        return valid_count == len(turn_env_states)

    def _check_prediction(self, response: str, turn_env_states: List[dict]) -> bool:
        """Verify <prediction> text matches actual post-action state."""
        pred_match = re.search(r"<prediction>(.*?)</prediction>", response, re.DOTALL)
        if not pred_match or not turn_env_states:
            return False
        pred_text = pred_match.group(1).lower()

        last_state = turn_env_states[-1]
        player_pos = last_state.get("player_pos")
        box_positions = last_state.get("box_positions", [])
        if not player_pos:
            return False

        described = self._parse_relative_position(pred_text, "box")
        if not described:
            return False

        for box_pos in box_positions:
            if self._verify_position(described, player_pos, box_pos):
                return True
        return False

    def _check_progress(self, turn_env_states: List[dict], initial_env_state: Optional[dict]) -> bool:
        """Check if boxes moved closer to targets overall."""
        if not turn_env_states:
            return False

        start_state = initial_env_state if initial_env_state else turn_env_states[0]
        end_state = turn_env_states[-1]

        start_bot = start_state.get("boxes_on_target", 0)
        end_bot = end_state.get("boxes_on_target", 0)
        if end_bot > start_bot:
            return True

        start_boxes = start_state.get("box_positions", [])
        end_boxes = end_state.get("box_positions", [])
        targets = end_state.get("target_positions", [])

        if not targets or not start_boxes or not end_boxes:
            return False

        start_dist = self._min_total_distance(start_boxes, targets)
        end_dist = self._min_total_distance(end_boxes, targets)
        return end_dist < start_dist

    def _parse_relative_position(self, text: str, obj_name: str) -> Optional[tuple]:
        """Extract (row_sign, col_sign) for an object from text."""
        pattern = rf"{obj_name}\s+is\s+(.*?)(?:of the player|of player)"
        match = re.search(pattern, text)
        if not match:
            pattern = rf"{obj_name}\s+(?:is|will be)\s+(.*?)(?:of the player|of player)"
            match = re.search(pattern, text)
        if not match:
            return None

        desc = match.group(1).lower()
        row_sign = None
        col_sign = None
        for word, sign in self.VERTICAL_MAP.items():
            if word in desc:
                row_sign = sign
                break
        for word, sign in self.HORIZONTAL_MAP.items():
            if word in desc:
                col_sign = sign
                break

        if row_sign is None and col_sign is None:
            return None
        return (row_sign, col_sign)

    @staticmethod
    def _verify_position(described: tuple, player_pos: list, object_pos: list) -> bool:
        """Check if described relative position matches actual coordinates."""
        row_sign, col_sign = described
        actual_row_diff = object_pos[0] - player_pos[0]
        actual_col_diff = object_pos[1] - player_pos[1]

        row_ok = True
        if row_sign is not None:
            if row_sign == 0:
                row_ok = actual_row_diff == 0
            else:
                row_ok = (actual_row_diff > 0) == (row_sign > 0) if actual_row_diff != 0 else False

        col_ok = True
        if col_sign is not None:
            if col_sign == 0:
                col_ok = actual_col_diff == 0
            else:
                col_ok = (actual_col_diff > 0) == (col_sign > 0) if actual_col_diff != 0 else False

        return row_ok and col_ok

    @staticmethod
    def _min_total_distance(boxes: list, targets: list) -> float:
        """Sum of each box's minimum Manhattan distance to any target."""
        total = 0.0
        for box in boxes:
            min_d = float("inf")
            for tgt in targets:
                d = abs(box[0] - tgt[0]) + abs(box[1] - tgt[1])
                if d < min_d:
                    min_d = d
            total += min_d
        return total


class SVGGroundedVerifier(BaseVerifier):
    """Verify SVG rubrics with cheap features independent of reward models.

    The policy rubricator historically emitted Sokoban rule IDs for every
    task.  During the transition to SVG-specific prompts we support both the
    explicit SVG IDs and the legacy four IDs, so alpha ablations are usable
    immediately and old checkpoints/configs remain debuggable.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = dict(cfg or {})
        self.thresholds = {
            "valid": float(cfg.get("valid_threshold", 0.5)),
            "layout": float(cfg.get("layout_threshold", 0.25)),
            "color": float(cfg.get("color_threshold", 0.45)),
            "edge": float(cfg.get("edge_threshold", 0.15)),
            "element": float(cfg.get("element_threshold", 0.50)),
        }

    def judge_many(
        self,
        criteria: List[str],
        question: str,
        response: str,
        cot: str,
        **kwargs,
    ) -> List[bool]:
        del question, response, cot
        extra_info = kwargs.get("extra_info", {}) or {}
        context = extra_info.get("verifier_context", {}) or {}
        rubrics = kwargs.get("rubrics", []) or []

        if not isinstance(context, dict):
            return [False] * len(criteria)

        results: List[bool] = []
        for idx in range(len(criteria)):
            rubric = rubrics[idx] if idx < len(rubrics) else None
            rule_id = str(getattr(rubric, "rule_id", "") or "").upper()
            results.append(self._judge_rule(rule_id, context))
        return results

    def _judge_rule(self, rule_id: str, context: dict) -> bool:
        valid = bool(context.get("valid_svg", False))
        if rule_id in {"SVG_VALIDITY", "ACTION_LEGALITY"}:
            return valid
        if not valid:
            return False

        mapping = {
            "LAYOUT_ALIGNMENT": ("layout_similarity", "layout"),
            "COMPOSITION_ALIGNMENT": ("layout_similarity", "layout"),
            "SPATIAL_GROUNDING": ("layout_similarity", "layout"),
            "COLOR_ALIGNMENT": ("color_similarity", "color"),
            "STYLE_ALIGNMENT": ("color_similarity", "color"),
            "LOGIC_FRESHNESS": ("color_similarity", "color"),
            "EDGE_STRUCTURE": ("edge_similarity", "edge"),
            "SHAPE_ALIGNMENT": ("edge_similarity", "edge"),
            "STRATEGIC_PROGRESS": ("edge_similarity", "edge"),
            "ELEMENT_STRUCTURE": ("element_similarity", "element"),
            "STRUCTURAL_ALIGNMENT": ("element_similarity", "element"),
            "PUSH_SIDE_POSITIONING": ("element_similarity", "element"),
            "PRE_TARGET_ALIGNMENT": ("layout_similarity", "layout"),
        }
        metric_and_threshold = mapping.get(rule_id)
        if metric_and_threshold is None:
            # An unknown semantic family has no grounded SVG interpretation.
            # Falling back to element similarity can silently accept unrelated
            # or negatively worded rubrics and corrupt acceptance statistics.
            return False
        metric_name, threshold_name = metric_and_threshold
        try:
            value = float(context.get(metric_name, 0.0))
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value >= self.thresholds[threshold_name])


def _build_verifier(verifier_cfg: dict) -> BaseVerifier:
    mode = str(verifier_cfg.get("mode", "heuristic")).lower()
    key = _make_cache_key(mode, verifier_cfg)
    if key in _VERIFIER_CACHE:
        return _VERIFIER_CACHE[key]

    if mode in {"heuristic", "rule"}:
        inst: BaseVerifier = HeuristicVerifier()
    elif mode == "grounded":
        inst = GroundedVerifier()
    elif mode == "svg_grounded":
        inst = SVGGroundedVerifier(verifier_cfg)
    else:
        inst = LLMVerifier(verifier_cfg)

    _VERIFIER_CACHE[key] = inst
    return inst


# ---------------------------------------------------------------------------
# EMA batch-level rule correlation tracker (feature: ema_rule_corr_filter)
# ---------------------------------------------------------------------------
# Maintains an exponential moving average of per-rule-id correlation with
# outcome across batches.  In n=1 PPO mode where per-group correlation is
# unavailable, this provides a statistical basis for filtering rubrics.
# ---------------------------------------------------------------------------

class RuleCorrTracker:
    """Exponentially decayed online correlation statistics per namespaced rule."""

    def __init__(self, ema_beta: float = 0.98):
        if not 0.0 <= float(ema_beta) < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        self.ema_beta = float(ema_beta)
        # key -> exponentially decayed raw moments and observed sample count.
        self._buffers: Dict[str, dict] = {}

    def update(self, rule_ids: List[str], sats: List[bool], outcome: float):
        """Update decayed moments for one grounded sample."""
        beta = self.ema_beta
        z = float(outcome)
        for rule_id, sat in zip(rule_ids, sats):
            if not rule_id:
                continue
            v = 1.0 if sat else 0.0
            buf = self._buffers.setdefault(
                rule_id,
                {"w": 0.0, "v": 0.0, "z": 0.0, "vz": 0.0, "v2": 0.0, "z2": 0.0, "count": 0},
            )
            for name in ("w", "v", "z", "vz", "v2", "z2"):
                buf[name] *= beta
            buf["w"] += 1.0
            buf["v"] += v
            buf["z"] += z
            buf["vz"] += v * z
            buf["v2"] += v * v
            buf["z2"] += z * z
            buf["count"] += 1

    def update_batch(
        self,
        observations: List[tuple[List[str], List[bool], float]],
    ) -> None:
        """Apply one order-invariant batch of grounded observations.

        Each calibration batch applies decay exactly once. Observations within
        the batch are aggregated without ordering, so ``ema_beta`` controls a
        history measured in calibration batches rather than being exhausted by
        one large batch of rubric observations.
        """
        by_rule: Dict[str, List[tuple[float, float]]] = {}
        for rule_ids, sats, outcome in observations:
            z = float(outcome)
            for rule_id, sat in zip(rule_ids, sats):
                if rule_id:
                    by_rule.setdefault(rule_id, []).append(
                        (1.0 if sat else 0.0, z)
                    )

        beta = self.ema_beta
        for rule_id, values in by_rule.items():
            n = len(values)
            if n == 0:
                continue
            buf = self._buffers.setdefault(
                rule_id,
                {"w": 0.0, "v": 0.0, "z": 0.0, "vz": 0.0, "v2": 0.0, "z2": 0.0, "count": 0},
            )
            for name in ("w", "v", "z", "vz", "v2", "z2"):
                buf[name] *= beta
            buf["w"] += n
            buf["v"] += sum(v for v, _ in values)
            buf["z"] += sum(z for _, z in values)
            buf["vz"] += sum(v * z for v, z in values)
            buf["v2"] += sum(v * v for v, _ in values)
            buf["z2"] += sum(z * z for _, z in values)
            buf["count"] += n

    def get_corr(self, rule_id: str, min_samples: int = 16) -> Optional[float]:
        """Return decayed correlation when sample mass and both variances suffice."""
        buf = self._buffers.get(rule_id)
        if buf is None or int(buf["count"]) < int(min_samples) or float(buf["w"]) < 2.0:
            return None
        w = float(buf["w"])
        mean_v = float(buf["v"]) / w
        mean_z = float(buf["z"]) / w
        var_v = float(buf["v2"]) / w - mean_v * mean_v
        var_z = float(buf["z2"]) / w - mean_z * mean_z
        if var_v <= 1e-12 or var_z <= 1e-12:
            return None
        cov = float(buf["vz"]) / w - mean_v * mean_z
        corr = cov / math.sqrt(var_v * var_z)
        if not math.isfinite(corr):
            return None
        return float(max(-1.0, min(1.0, corr)))

    def get_diagnostics(self, rule_id: str, min_samples: int = 16) -> Dict[str, float]:
        """Return numeric tracker state suitable for reward/trainer diagnostics."""
        buf = self._buffers.get(rule_id)
        if buf is None:
            return {"count": 0.0, "effective_n": 0.0, "corr": 0.0, "corr_available": 0.0}
        corr = self.get_corr(rule_id, min_samples=min_samples)
        return {
            "count": float(buf["count"]),
            "effective_n": float(buf["w"]),
            "corr": float(corr) if corr is not None else 0.0,
            "corr_available": float(corr is not None),
        }

    def clear(self):
        self._buffers.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_beta": float(self.ema_beta),
            "buffers": {
                key: dict(values) for key, values in self._buffers.items()
            },
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        self.clear()
        if not isinstance(state, dict):
            return
        if "ema_beta" in state:
            self.ema_beta = float(state["ema_beta"])
        buffers = state.get("buffers", {})
        if not isinstance(buffers, dict):
            return
        for key, values in buffers.items():
            if not isinstance(values, dict):
                continue
            self._buffers[str(key)] = {
                "w": float(values.get("w", 0.0)),
                "v": float(values.get("v", 0.0)),
                "z": float(values.get("z", 0.0)),
                "vz": float(values.get("vz", 0.0)),
                "v2": float(values.get("v2", 0.0)),
                "z2": float(values.get("z2", 0.0)),
                "count": int(values.get("count", 0)),
            }


# Global singleton — persists across batches within a training run.
_rule_corr_tracker = RuleCorrTracker()


def _corr_binary(v: np.ndarray, z: np.ndarray) -> float:
    if v.size < 2 or z.size < 2:
        return 0.0
    if float(np.std(v)) <= 1e-12 or float(np.std(z)) <= 1e-12:
        return 0.0
    c = np.corrcoef(v.astype(float), z.astype(float))[0, 1]
    if np.isnan(c) or np.isinf(c):
        return 0.0
    return float(c)


def _outcome_reward(data_source: Any, solution_str: str, ground_truth: Any, extra_info: dict) -> float:
    # Prefer the environment reward from the agent loop (rm_scores) when available,
    # since default_compute_score only handles built-in data sources (GSM8K, MATH, etc.)
    # and will raise NotImplementedError for custom environments like Sokoban.
    env_reward = extra_info.get("env_reward", None)
    if env_reward is not None:
        return float(env_reward)
    try:
        r = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(r, dict):
            val = float(r.get("score", 0.0))
        else:
            val = float(r)
    except Exception:
        val = 0.0
    return float(val)


def _alignment_target(
    data_source: Any,
    extra_info: dict,
    outcome: float,
    target_name: str = "auto",
) -> float:
    """Resolve the grounded target z used for rubric correlation.

    Discrete environments should explicitly use ``traj_success`` so format
    shaping rewards do not turn failed trajectories into positive outcomes.
    Continuous tasks such as SVG should use ``env_reward`` (or a named visual
    metric). ``auto`` preserves a safe generic behavior: prefer trajectory
    success when present, otherwise use the environment outcome.
    """
    target_name = str(target_name).lower()
    if target_name == "auto":
        source_name = str(data_source).lower()
        # SVG is a continuous-quality task; its binary success threshold is a
        # logging aid, not the primary environment-alignment target.
        if "svg" not in source_name and "traj_success" in extra_info:
            return float(extra_info["traj_success"])
        return float(outcome)
    if target_name in {"outcome", "env_reward"}:
        if target_name == "env_reward" and extra_info.get("env_reward") is not None:
            return float(extra_info["env_reward"])
        return float(outcome)
    if target_name == "visual_similarity":
        vals = [
            float(extra_info[key])
            for key in ("dino_score", "dreamsim_score")
            if extra_info.get(key) is not None
        ]
        return float(np.mean(vals)) if vals else float(outcome)
    if target_name not in extra_info:
        raise KeyError(
            f"alignment_target={target_name!r} is unavailable for data_source={data_source!r}; "
            f"available keys={sorted(extra_info.keys())}"
        )
    return float(extra_info[target_name])


def _validate_rubric_correlations(
    v_matrix: np.ndarray,
    z: np.ndarray,
    alpha: float,
    rule_ids: List[str],
    ema_rule_corr_filter: bool,
    insufficient_corr_policy: str,
    corr_min_samples: int = 16,
) -> tuple[List[bool], List[Optional[float]], bool]:
    """Validate rubrics using Pearson alignment with grounded feedback.

    Returns ``(valid_flags, correlations, group_target_is_computable)``.
    Correlation is used only when both satisfaction and grounded targets vary.
    When group-level evidence is unavailable, the configured policy is applied
    unless a historical rule correlation is available.
    """
    policy = str(insufficient_corr_policy).lower()
    if policy not in {"accept", "reject"}:
        raise ValueError(
            "insufficient_corr_policy must be 'accept' or 'reject', "
            f"got {insufficient_corr_policy!r}"
        )

    n_rules = int(v_matrix.shape[0])
    group_computable = z.size >= 2 and float(np.std(z)) > 1e-12
    valid_flags: List[bool] = []
    correlations: List[Optional[float]] = []

    if group_computable:
        for k in range(n_rules):
            vk = v_matrix[k, :]
            if float(np.std(vk)) <= 1e-12:
                correlations.append(None)
                valid_flags.append(False)
                continue
            corr = _corr_binary(vk, z)
            correlations.append(corr)
            valid_flags.append(bool(corr > alpha))
        return valid_flags, correlations, True

    default_flag = policy == "accept"
    for k in range(n_rules):
        corr: Optional[float] = None
        if ema_rule_corr_filter:
            rid = rule_ids[k] if k < len(rule_ids) else ""
            if rid:
                corr = _rule_corr_tracker.get_corr(rid, min_samples=corr_min_samples)
        correlations.append(corr)
        valid_flags.append(bool(corr > alpha) if corr is not None else default_flag)
    return valid_flags, correlations, False


def _norm_minmax(agg: float, min_v: float, max_v: float) -> float:
    if max_v - min_v <= 1e-12:
        return 0.0
    x = (agg - min_v) / (max_v - min_v)
    return float(max(0.0, min(1.0, x)))


def _score_rubrics(
    points: List[float],
    satisfied: List[bool],
    selected_indices: List[int],
) -> float:
    """Return the normalized rubric score for a selected rubric subset."""
    if not selected_indices:
        return 0.0
    agg = 0.0
    min_v = 0.0
    max_v = 0.0
    for k in selected_indices:
        point = float(points[k])
        if bool(satisfied[k]):
            agg += point
        max_v += max(0.0, point)
        min_v += min(0.0, point)
    return _norm_minmax(agg, min_v, max_v)


def _rubric_decision_reason(
    *,
    correlation: Optional[float],
    correlation_accepted: bool,
    target_is_computable: bool,
    satisfaction_is_constant: bool,
) -> str:
    """Produce an explicit, reviewer-facing reason for a validation decision."""
    if correlation_accepted:
        return "corr_above_alpha"
    if not target_is_computable:
        return "constant_outcome_or_insufficient_group"
    if satisfaction_is_constant:
        return "constant_satisfaction"
    if correlation is None:
        return "insufficient_statistical_evidence"
    if float(correlation) < 0.0:
        return "negative_correlation"
    return "corr_below_alpha"


def _group_key(extra_info: dict, fallback_idx: int) -> str:
    for key in ("group_idx", "uid", "question_id", "traj_uid"):
        if key in extra_info and extra_info[key] is not None:
            return str(extra_info[key])
    return f"__single__{fallback_idx}"


def _question_text(extra_info: dict) -> str:
    for key in ("question", "query", "prompt_str", "prompt", "instruction"):
        v = extra_info.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _compute_score_batched(
    *,
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    alpha: float = 0.2,
    lambda_cot: float = 0.0,
    outcome_weight: float = 1.0,
    pure_outcome: bool = False,
    rubricator: Optional[dict] = None,
    verifier: Optional[dict] = None,
    fallback_to_heuristic: bool = True,
    normalize_rubric_weights: bool = True,
    ema_rule_corr_filter: bool = True,
    insufficient_corr_policy: str = "reject",
    alignment_target: str = "auto",
    trivial_rubric_cooldown: bool = True,
    trivial_rubric_scale: float = 0.5,
    alignment_rubricator_reward: bool = True,
    rubric_validation_enabled: bool = True,
    rubric_audit: Optional[dict] = None,
    corr_ema_beta: float = 0.98,
    corr_min_samples: int = 16,
    corr_calibration_interval: int = 10,
    corr_schema_version: str = "v1",
    **kwargs,
) -> List[dict]:
    # Feature flags for backward compatibility:
    # - normalize_rubric_weights: clamp + L1-normalize rubric weights (default True)
    # - ema_rule_corr_filter: use historical rule correlation when group evidence is unavailable
    # - insufficient_corr_policy: reject (strict) or accept rubrics without statistical evidence
    # - alignment_target: grounded feedback z used for correlation
    # - trivial_rubric_cooldown: scale cot_reward by 0.1 when all-SAT/all-UNSAT (default True)
    # - alignment_rubricator_reward: align rubricator reward with outcome (default True)
    # - rubric_validation_enabled: explicit unfiltered-shaping control (default True)
    # - rubric_audit: sparse structured logging only; never changes rewards

    # Pure outcome mode: skip all rubric computation, return only environment reward.
    if pure_outcome:
        outputs: List[dict] = []
        for i in range(len(solution_strs)):
            extra = dict(extra_infos[i] or {})
            outcome = _outcome_reward(data_sources[i], solution_strs[i], ground_truths[i], extra)
            outputs.append({
                "score": float(outcome),
                "outcome_reward": float(outcome),
                "cot_reward": 0.0,
                "rubricator_reward": 0.0,
                "valid_ratio": 0.0,
                "corr_mean": 0.0,
                "trivial_scale": 1.0,
            })
        return outputs

    alpha = float(alpha)
    lambda_cot = float(lambda_cot)
    outcome_weight = float(outcome_weight)
    trivial_rubric_scale = float(
        max(0.0, min(1.0, float(trivial_rubric_scale)))
    )
    rubricator_cfg = rubricator or {}
    verifier_cfg = verifier or {}
    audit_cfg = dict(rubric_audit or {})
    audit_enabled = bool(audit_cfg.get("enabled", False))
    audit_interval = max(1, int(audit_cfg.get("interval", 20)))
    audit_max_groups = max(1, int(audit_cfg.get("max_groups", 2)))
    audit_include_raw = bool(audit_cfg.get("include_raw_proposal", True))
    audit_max_raw_chars = max(0, int(audit_cfg.get("max_raw_chars", 6000)))
    _rule_corr_tracker.ema_beta = float(corr_ema_beta)
    corr_min_samples = max(2, int(corr_min_samples))
    corr_calibration_interval = max(1, int(corr_calibration_interval))
    rubricator_mode = str(rubricator_cfg.get("mode", "policy")).lower()
    policy_max_rubrics = int(rubricator_cfg.get("max_rubrics", 8))
    
    rub = _build_rubricator(rubricator_cfg)
    ver = _build_verifier(verifier_cfg)
    
    n = len(solution_strs)
    items: List[Dict[str, Any]] = []
    for i in range(n):
        sol = str(solution_strs[i])
        extra = dict(extra_infos[i] or {})
        cot, final_answer = _extract_cot_and_answer(sol)
        q = _question_text(extra)
        gkey = _group_key(extra, i)
        traj_idx = extra.get("traj_idx", None)
        turn_idx = _to_int_or_none(extra.get("turn_idx", None))
        prompt_str = extra.get("prompt_str", "")

        outcome = _outcome_reward(data_sources[i], sol, ground_truths[i], extra)
        target = _alignment_target(
            data_source=data_sources[i],
            extra_info=extra,
            outcome=outcome,
            target_name=alignment_target,
        )

        items.append(
            {
                "idx": i,
                "group": gkey,
                "question": q,
                "solution": sol,
                "cot": cot,
                "answer": final_answer,
                "traj_idx": traj_idx,
                "turn_idx": turn_idx,
                "prompt_str": prompt_str,
                "proposal": None,
                "proposal_source": None,
                "policy_proposal_raw": extra.get("rlcer_policy_rubric_raw", None),
                "outcome": outcome,
                "alignment_target": target,
                "extra_info": extra,
            }
        )

    # Build trajectory histories first, then resolve rubric proposals.
    # When rubricator_mode == "policy", precomputed rubrics from the training
    # loop (ray_trainer.py) are injected via batch.non_tensor_batch and passed
    # through as extra_info["rlcer_policy_rubric_raw"].  We prefer these and
    # never call LLMRubricator again — the policy endpoint is already busy
    # serving rollout requests.  Missing / unparseable precomputed rubrics
    # fall back to HeuristicRubricator instead of a redundant LLM call.
    histories = _build_trajectory_history_for_items(items)
    for i in range(n):
        # 1) Try precomputed policy rubrics (from ray_trainer.py).
        
        if rubricator_mode == "policy":
            precomputed = _extract_policy_rubric_proposal_from_extra(
                extra_info=items[i].get("extra_info", {}),
                max_rubrics=policy_max_rubrics,
            )

            if precomputed is not None and precomputed.format_ok and len(precomputed.rubrics) > 0:
                items[i]["proposal"] = precomputed
                items[i]["proposal_source"] = "policy"
                continue
            # Precomputed not available / unparseable -> heuristic fallback.
            if fallback_to_heuristic:
                try:
                    items[i]["proposal"] = HeuristicRubricator(max_rubrics=policy_max_rubrics).generate(
                        question=items[i]["question"],
                        response=items[i]["solution"],
                        cot=items[i]["cot"],
                    )
                    items[i]["proposal_source"] = "heuristic_fallback"
                except Exception:
                    items[i]["proposal"] = RubricProposal(rubrics=[], format_ok=False, raw="heuristic_error")
                    items[i]["proposal_source"] = "heuristic_error"
                continue
            else:
                items[i]["proposal"] = RubricProposal(rubrics=[], format_ok=False, raw="no_precomputed_rubric")
                items[i]["proposal_source"] = "missing_policy_proposal"
                continue

        # 2) Non-policy modes (heuristic / external) use the configured rubricator.
        try:
            items[i]["proposal"] = rub.generate(
                question=items[i]["question"],
                response=items[i]["solution"],
                cot=items[i]["cot"],
                trajectory_history=histories[i],
                turn_idx=items[i].get("turn_idx", None),
                extra_info=items[i].get("extra_info", {}),
            )
            items[i]["proposal_source"] = rubricator_mode
        except Exception:
            if not fallback_to_heuristic:
                items[i]["proposal"] = RubricProposal(rubrics=[], format_ok=False, raw="rubricator_error")
                items[i]["proposal_source"] = f"{rubricator_mode}_error"
            else:
                items[i]["proposal"] = HeuristicRubricator().generate(
                    question=items[i]["question"],
                    response=items[i]["solution"],
                    cot=items[i]["cot"],
                )
                items[i]["proposal_source"] = "heuristic_fallback"

    group_to_indices: Dict[str, List[int]] = {}
    for i, x in enumerate(items):
        group_to_indices.setdefault(x["group"], []).append(i)

    audit_step = _to_int_or_none(
        items[0]["extra_info"].get("rlcer_global_step") if items else None
    )
    audit_this_step = bool(
        audit_enabled
        and audit_step is not None
        and (audit_step == 1 or audit_step % audit_interval == 0)
    )
    audited_groups = (
        set(list(group_to_indices.keys())[:audit_max_groups])
        if audit_this_step
        else set()
    )

    cot_rewards = [0.0] * n
    all_rubric_rewards = [0.0] * n
    rubric_audits: List[Optional[dict]] = [None] * n
    trivial_scales = [1.0] * n  # all-SAT/all-UNSAT cooldown factor per sample
    valid_ratios = [0.0] * n
    evolving_rewards = [0.0] * n
    corr_means = [0.0] * n
    corr_means_all = [0.0] * n
    accepted_corr_means = [0.0] * n
    rejected_corr_means = [0.0] * n
    accepted_corr_counts = [0] * n
    rejected_corr_counts = [0] * n
    rubric_counts = [0] * n
    correlation_computable_counts = [0] * n
    correlation_unavailable_counts = [0] * n
    conditional_acceptance_rates = [0.0] * n
    correlation_unavailable_rates = [0.0] * n
    corr_computable_ratios = [0.0] * n
    group_corr_used = [0.0] * n
    tracker_corr_used = [0.0] * n
    tracker_corr_values = [0.0] * n
    tracker_effective_ns = [0.0] * n
    tracker_observation_counts = [0.0] * n
    tracker_keys_by_item: List[List[str]] = [[] for _ in range(n)]
    pending_tracker_updates: List[tuple[List[str], List[bool], float]] = []
    group_target_computable = [0.0] * n
    group_sizes = [0] * n
    alignment_target_stds = [0.0] * n

    for _, idxs in group_to_indices.items():
        z = np.array([items[t]["alignment_target"] for t in idxs], dtype=float)
        group_corrs: List[float] = []
        target_std = float(np.std(z))
        target_is_computable = len(idxs) >= 2 and target_std > 1e-12
        for i in idxs:
            group_target_computable[i] = float(target_is_computable)
            group_sizes[i] = len(idxs)
            alignment_target_stds[i] = target_std

        def _normalize_judged(judged: List[bool], criterion_count: int) -> List[bool]:
            out = [bool(x) for x in judged[:criterion_count]]
            if len(out) < criterion_count:
                out.extend([False] * (criterion_count - len(out)))
            return out

        def _judge_for_rubrics(j: int, criteria: List[str], rubrics: List[Rubric]) -> List[bool]:
            try:
                judged = ver.judge_many(
                    criteria=criteria,
                    question=items[j]["question"],
                    response=items[j]["solution"],
                    cot=items[j]["cot"],
                    rubrics=rubrics,
                    extra_info=items[j]["extra_info"],
                )
            except Exception:
                if fallback_to_heuristic:
                    judged = HeuristicVerifier().judge_many(
                        criteria=criteria,
                        question=items[j]["question"],
                        response=items[j]["solution"],
                        cot=items[j]["cot"],
                        rubrics=rubrics,
                        extra_info=items[j]["extra_info"],
                    )
                else:
                    judged = [False] * len(criteria)
            return _normalize_judged(judged, len(criteria))

        # A policy can propose a different rubric set for every trajectory.
        # For n=8, evaluating each proposal over its group is O(n² × rubrics).
        # All of these VLM calls are independent, so prefetch the entire group
        # through a bounded pool instead of waiting on thousands of serial HTTP
        # round trips. The trained verifier's default cap (16) stays below the
        # SGLang service's max-running-requests setting (32).
        prefetched_judgements: Dict[tuple[int, int], List[bool]] = {}
        request_workers = min(
            getattr(ver, "request_concurrency", 1),
            sum(len(items[src]["proposal"].rubrics) > 0 for src in idxs) * len(idxs),
        )
        if request_workers > 1:
            prefetch_jobs: List[tuple[int, int, List[str], List[Rubric]]] = []
            for src in idxs:
                source_rubrics = items[src]["proposal"].rubrics
                if source_rubrics:
                    source_criteria = [r.criterion for r in source_rubrics]
                    prefetch_jobs.extend(
                        (src, j, source_criteria, source_rubrics) for j in idxs
                    )

            def _prefetch(job: tuple[int, int, List[str], List[Rubric]]) -> tuple[tuple[int, int], List[bool]]:
                src, j, source_criteria, source_rubrics = job
                return (src, j), _judge_for_rubrics(j, source_criteria, source_rubrics)

            with ThreadPoolExecutor(max_workers=request_workers) as executor:
                for key, judged in executor.map(_prefetch, prefetch_jobs):
                    prefetched_judgements[key] = judged

        for i in idxs:
            proposal: RubricProposal = items[i]["proposal"]
            rubrics = proposal.rubrics
            if len(rubrics) == 0:
                cot_rewards[i] = 0.0
                valid_ratios[i] = 0.0
                evolving_rewards[i] = 1.0 if proposal.format_ok else 0.0
                if items[i]["group"] in audited_groups:
                    raw_proposal = str(
                        items[i].get("policy_proposal_raw")
                        or getattr(proposal, "raw", "")
                        or ""
                    )
                    if not audit_include_raw:
                        raw_proposal = ""
                    elif audit_max_raw_chars:
                        raw_proposal = raw_proposal[:audit_max_raw_chars]
                    rubric_audits[i] = {
                        "step": int(audit_step),
                        "sample_index": int(i),
                        "group": str(items[i]["group"]),
                        "traj_idx": (
                            None
                            if items[i].get("traj_idx") is None
                            else int(items[i]["traj_idx"])
                        ),
                        "proposal_source": str(items[i].get("proposal_source") or "unknown"),
                        "proposal_format_ok": bool(proposal.format_ok),
                        "raw_proposal": raw_proposal,
                        "alpha": float(alpha),
                        "rubric_validation_enabled": bool(rubric_validation_enabled),
                        "alignment_target": float(items[i]["alignment_target"]),
                        "group_outcomes": [float(x) for x in z.tolist()],
                        "group_target_computable": bool(target_is_computable),
                        "validated_rubric_score": 0.0,
                        "all_rubric_score": 0.0,
                        "rubrics": [],
                    }
                continue

            criteria = [r.criterion for r in rubrics]
            points = [float(r.points) for r in rubrics]

            # evaluate each criterion over all trajectories in same question-group
            v_matrix = np.zeros((len(criteria), len(idxs)), dtype=float)
            if prefetched_judgements:
                judged_by_pos = [prefetched_judgements[(i, j)] for j in idxs]
            else:
                workers = min(len(idxs), getattr(ver, "group_concurrency", 1))
                if workers > 1:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        judged_by_pos = list(executor.map(
                            lambda j: _judge_for_rubrics(j, criteria, rubrics), idxs
                        ))
                else:
                    judged_by_pos = [_judge_for_rubrics(j, criteria, rubrics) for j in idxs]

            for j_pos, judged in enumerate(judged_by_pos):
                v_matrix[:, j_pos] = np.array([1.0 if b else 0.0 for b in judged], dtype=float)


            i_pos = idxs.index(i)
            sat_self = [bool(v_matrix[k, i_pos] > 0.5) for k in range(len(criteria))]

            # --- Validity filtering: per-group corr or EMA batch corr ---
            rule_ids = [canonicalize_rule_id(r.rule_id) for r in rubrics]
            tracker_keys = [
                (
                    _rule_corr_key(
                        data_sources[i],
                        rid,
                        schema_version=str(corr_schema_version),
                    )
                    if rid
                    else ""
                )
                for rid in rule_ids
            ]
            tracker_keys_by_item[i] = tracker_keys
            correlation_flags, correlations, target_is_computable = _validate_rubric_correlations(
                v_matrix=v_matrix,
                z=z,
                alpha=alpha,
                rule_ids=tracker_keys,
                ema_rule_corr_filter=ema_rule_corr_filter,
                insufficient_corr_policy=insufficient_corr_policy,
                corr_min_samples=corr_min_samples,
            )
            valid_flags = (
                list(correlation_flags)
                if rubric_validation_enabled
                else [True] * len(criteria)
            )
            numeric_corrs = [float(c) for c in correlations if c is not None]
            valid_corrs = [
                float(c)
                for c, is_valid in zip(correlations, correlation_flags, strict=True)
                if c is not None and is_valid
            ]
            rejected_corrs = [
                float(c)
                for c, is_valid in zip(correlations, correlation_flags, strict=True)
                if c is not None and not is_valid
            ]
            group_corrs.extend(valid_corrs)
            accepted_corr_means[i] = (
                float(np.mean(valid_corrs)) if valid_corrs else 0.0
            )
            rejected_corr_means[i] = (
                float(np.mean(rejected_corrs)) if rejected_corrs else 0.0
            )
            accepted_corr_counts[i] = len(valid_corrs)
            rejected_corr_counts[i] = len(rejected_corrs)
            rubric_counts[i] = len(criteria)
            correlation_computable_counts[i] = len(numeric_corrs)
            correlation_unavailable_counts[i] = len(criteria) - len(numeric_corrs)
            conditional_acceptance_rates[i] = (
                len(valid_corrs) / len(numeric_corrs) if numeric_corrs else 0.0
            )
            correlation_unavailable_rates[i] = (
                (len(criteria) - len(numeric_corrs)) / len(criteria)
                if criteria
                else 0.0
            )
            corr_means_all[i] = float(np.mean(numeric_corrs)) if numeric_corrs else 0.0
            corr_computable_ratios[i] = (
                float(len(numeric_corrs) / len(criteria)) if criteria else 0.0
            )
            group_corr_used[i] = float(target_is_computable and bool(numeric_corrs))
            tracker_corr_used[i] = float((not target_is_computable) and bool(numeric_corrs))
            if not target_is_computable and numeric_corrs:
                tracker_corr_values[i] = float(np.mean(numeric_corrs))
            valid_idx = [k for k, flag in enumerate(valid_flags) if flag]
            k_total = len(criteria)
            k_valid = len(valid_idx)
            valid_ratio = (k_valid / k_total) if k_total > 0 else 0.0
            valid_ratios[i] = float(valid_ratio)

            # --- All-SAT / All-UNSAT cooldown ---
            # If verifier judges ALL criteria as SAT (or all UNSAT), the rubric
            # is likely trivial or uninformative.  Scale cot_reward down.
            if trivial_rubric_cooldown:
                all_sat = all(sat_self) if sat_self else False
                all_unsat = (not any(sat_self)) if sat_self else False
                trivial_scale = (
                    trivial_rubric_scale if (all_sat or all_unsat) else 1.0
                )
            else:
                trivial_scale = 1.0
            trivial_scales[i] = trivial_scale

            # --- Compute validated and unfiltered counterfactual scores ---
            format_reward = 1.0 if proposal.format_ok else 0.0
            all_raw_cot = _score_rubrics(
                points=points,
                satisfied=sat_self,
                selected_indices=list(range(k_total)),
            )
            raw_cot = _score_rubrics(
                points=points,
                satisfied=sat_self,
                selected_indices=valid_idx,
            )
            all_rubric_rewards[i] = all_raw_cot * trivial_scale
            cot_rewards[i] = raw_cot * trivial_scale

            if k_valid == 0:
                if alignment_rubricator_reward:
                    evolving_rewards[i] = float(format_reward)
                else:
                    evolving_rewards[i] = float(valid_ratio + format_reward)
            else:
                # --- Alignment-based rubricator reward ---
                if alignment_rubricator_reward:
                    target = items[i]["alignment_target"]
                    if target > 0:
                        alignment = raw_cot      # success: reward rubric for giving high score
                    else:
                        alignment = 1.0 - raw_cot  # failure: reward rubric for giving low score
                    evolving_rewards[i] = float(format_reward + valid_ratio * alignment)
                else:
                    evolving_rewards[i] = float(valid_ratio + format_reward)

            # --- Stage tracker observations for a batch-level update ---
            # Correlation validation for every item in this invocation must see
            # exactly the same pre-batch tracker state.  Applying updates here
            # would make later items depend on the iteration order.
            global_step = _to_int_or_none(items[i]["extra_info"].get("rlcer_global_step"))
            calibration_step = global_step is None or global_step % corr_calibration_interval == 0
            if ema_rule_corr_filter and calibration_step:
                pending_tracker_updates.append(
                    (tracker_keys, sat_self, float(items[i]["alignment_target"]))
                )

            if items[i]["group"] in audited_groups:
                raw_proposal = str(
                    items[i].get("policy_proposal_raw")
                    or getattr(proposal, "raw", "")
                    or ""
                )
                if not audit_include_raw:
                    raw_proposal = ""
                elif audit_max_raw_chars:
                    raw_proposal = raw_proposal[:audit_max_raw_chars]

                rubric_rows = []
                for k, rubric_obj in enumerate(rubrics):
                    corr = correlations[k]
                    corr_flag = bool(correlation_flags[k])
                    reason = _rubric_decision_reason(
                        correlation=None if corr is None else float(corr),
                        correlation_accepted=corr_flag,
                        target_is_computable=bool(target_is_computable),
                        satisfaction_is_constant=bool(
                            float(np.std(v_matrix[k, :])) <= 1e-12
                        ),
                    )
                    rubric_rows.append(
                        {
                            "rule_id": str(rubric_obj.rule_id),
                            "criterion": str(rubric_obj.criterion),
                            "weight": float(points[k]),
                            "satisfied_self": bool(sat_self[k]),
                            "satisfaction_vector": [
                                int(x) for x in v_matrix[k, :].tolist()
                            ],
                            "correlation": None if corr is None else float(corr),
                            "correlation_accepted": corr_flag,
                            "applied_accepted": bool(valid_flags[k]),
                            "decision_reason": (
                                reason
                                if rubric_validation_enabled
                                else "validation_disabled"
                            ),
                        }
                    )

                rubric_audits[i] = {
                    "step": int(audit_step),
                    "sample_index": int(i),
                    "group": str(items[i]["group"]),
                    "traj_idx": (
                        None
                        if items[i].get("traj_idx") is None
                        else int(items[i]["traj_idx"])
                    ),
                    "proposal_source": str(items[i].get("proposal_source") or "unknown"),
                    "proposal_format_ok": bool(proposal.format_ok),
                    "raw_proposal": raw_proposal,
                    "alpha": float(alpha),
                    "rubric_validation_enabled": bool(rubric_validation_enabled),
                    "alignment_target": float(items[i]["alignment_target"]),
                    "group_outcomes": [float(x) for x in z.tolist()],
                    "group_target_computable": bool(target_is_computable),
                    "validated_rubric_score": float(cot_rewards[i]),
                    "all_rubric_score": float(all_rubric_rewards[i]),
                    "rubrics": rubric_rows,
                }

        group_corr_mean = float(np.mean(group_corrs)) if group_corrs else 0.0
        for i in idxs:
            corr_means[i] = group_corr_mean

    # Commit all calibration observations only after the entire batch has been
    # validated.  This removes within-batch information leakage and makes the
    # result invariant to sample iteration order.
    _rule_corr_tracker.update_batch(pending_tracker_updates)

    # Keep diagnostics as post-batch state so monitoring still reflects the
    # observations committed by this calibration batch.
    for i, tracker_keys in enumerate(tracker_keys_by_item):
        tracker_diags = [
            _rule_corr_tracker.get_diagnostics(key, min_samples=corr_min_samples)
            for key in tracker_keys
            if key
        ]
        if tracker_diags:
            tracker_effective_ns[i] = float(
                np.mean([diag["effective_n"] for diag in tracker_diags])
            )
            tracker_observation_counts[i] = float(
                np.mean([diag["count"] for diag in tracker_diags])
            )

    outputs: List[dict] = []
    for i in range(n):
        outcome = float(items[i]["outcome"])
        r_cot = float(cot_rewards[i])
        score = float(outcome_weight * outcome + lambda_cot * r_cot)

        output = {
            "score": score,
            "outcome_reward": outcome,
            "cot_reward": r_cot,
            "rubricator_reward": float(evolving_rewards[i]),
            "valid_ratio": float(valid_ratios[i]),
            "rubric_accepted_rate": float(valid_ratios[i]),
            "rubric_acceptance_rate_overall": float(valid_ratios[i]),
            "rubric_acceptance_rate_given_computable": float(
                conditional_acceptance_rates[i]
            ),
            "correlation_unavailable_rate": float(
                correlation_unavailable_rates[i]
            ),
            "rubric_total_count": int(rubric_counts[i]),
            "correlation_computable_count": int(
                correlation_computable_counts[i]
            ),
            "correlation_unavailable_count": int(
                correlation_unavailable_counts[i]
            ),
            "corr_mean": float(corr_means[i]),
            "corr_mean_all": float(corr_means_all[i]),
            "mean_accepted_correlation": float(accepted_corr_means[i]),
            "mean_rejected_correlation": float(rejected_corr_means[i]),
            "accepted_correlation_count": int(accepted_corr_counts[i]),
            "rejected_correlation_count": int(rejected_corr_counts[i]),
            "corr_computable_ratio": float(corr_computable_ratios[i]),
            "group_corr_used": float(group_corr_used[i]),
            "tracker_corr_used": float(tracker_corr_used[i]),
            "tracker_corr": float(tracker_corr_values[i]),
            "tracker_effective_n": float(tracker_effective_ns[i]),
            "tracker_observation_count": float(tracker_observation_counts[i]),
            "group_target_computable": float(group_target_computable[i]),
            "group_size": int(group_sizes[i]),
            "alignment_target": float(items[i]["alignment_target"]),
            "alignment_target_std": float(alignment_target_stds[i]),
            "alpha": float(alpha),
            "trivial_scale": float(trivial_scales[i]),
            # Role-health signals must remain separate from rubric validation.
            # A zero valid_ratio can mean either that valid rubrics were rejected
            # or that the policy failed to emit a parseable/non-empty proposal.
            "proposal_format_ok": float(bool(items[i]["proposal"].format_ok)),
            "proposal_nonempty": float(bool(items[i]["proposal"].rubrics)),
            "reasoner_cot_format_ok": float(
                bool(re.search(r"<think>\s*.+?\s*</think>", solution_strs[i], re.DOTALL | re.IGNORECASE))
            ),
            "reasoner_answer_format_ok": float(
                bool(re.search(r"<answer>\s*.+?\s*</answer>", solution_strs[i], re.DOTALL | re.IGNORECASE))
            ),
        }
        if audit_enabled:
            # Empty dictionaries keep validation aggregation from treating sparse
            # audit rows as numeric metrics.
            output["all_rubric_score"] = float(all_rubric_rewards[i])
            output["rubric_audit"] = rubric_audits[i] or {}
        outputs.append(output)
    return outputs


def _compute_score_single(
    *,
    data_source,
    solution_str,
    ground_truth,
    extra_info,
    **kwargs,
):
    out = _compute_score_batched(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info or {}],
        **kwargs,
    )
    return out[0]


def compute_score(
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    """RLCER custom reward function.
    Supports both VERL batch reward-manager signature and single-item signature.

    Batch signature (recommended):
      compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)

    Single signature:
      compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    """
    
    if data_sources is not None and solution_strs is not None:
        return _compute_score_batched(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos or [{} for _ in range(len(solution_strs))],
            **kwargs,
        )

    return _compute_score_single(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info or {},
        **kwargs,
    )


def _get_rule_corr_tracker_state() -> Dict[str, Any]:
    return _rule_corr_tracker.state_dict()


def _load_rule_corr_tracker_state(state: Optional[Dict[str, Any]]) -> None:
    _rule_corr_tracker.load_state_dict(state)


# Reward managers retain the custom compute callable, so these hooks let the
# trainer persist process-local tracker statistics alongside PPO checkpoints.
compute_score.get_tracker_state = _get_rule_corr_tracker_state
compute_score.load_tracker_state = _load_rule_corr_tracker_state
