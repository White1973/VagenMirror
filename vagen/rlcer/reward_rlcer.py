from __future__ import annotations

import json
import math
import re
import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from verl.utils.reward_score import default_compute_score
from vagen.rlcer.prompt_templates import NEW_SYSTEM_CONTENT, USER_CONTENT


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


@dataclass
class Rubric:
    criterion: str
    points: float


@dataclass
class RubricProposal:
    rubrics: List[Rubric]
    format_ok: bool
    raw: str = ""


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

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _parse_rubric_proposal_from_raw(raw: str, max_rubrics: int) -> RubricProposal:
    obj = _safe_json_extract(raw)
    if not isinstance(obj, dict):
        return RubricProposal(rubrics=[], format_ok=False, raw=raw)

    rubrics_raw = obj.get("rubrics", [])
    rubrics: List[Rubric] = []
    for item in rubrics_raw:
        if not isinstance(item, dict):
            continue
        # Support both legacy and new schema.
        crit = str(item.get("criterion", item.get("description", ""))).strip()
        pts = item.get("points", item.get("weight", 0))
        if not crit:
            continue
        try:
            pts = float(pts)
        except Exception:
            continue
        if pts == 0:
            continue
        rubrics.append(Rubric(criterion=crit, points=pts))

    return RubricProposal(rubrics=rubrics[: max_rubrics], format_ok=len(rubrics) > 0, raw=raw)


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
            Rubric("Reasoning steps are coherent and logically connected", 3),
            Rubric("Solution includes at least one explicit verification/check", 2),
            Rubric("Reasoning stays focused on the target question", 2),
            Rubric("Reasoning contains arithmetic/algebraic inconsistency", -3),
            Rubric("Reasoning is repetitive without adding new progress", -2),
            Rubric("Final conclusion is clearly stated", 2),
        ][: self.max_rubrics]
        return RubricProposal(rubrics=rubrics, format_ok=True, raw="heuristic")


class HeuristicVerifier(BaseVerifier):
    """Heuristic verifier with deterministic lexical checks.

    This is intentionally simple and should be replaced by an external verifier model
    for high-fidelity judgement.
    """

    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        out: List[bool] = []
        cot_tokens = _normalize_token_set(cot)
        for c in criteria:
            crit = c or ""
            crit_tokens = _normalize_token_set(crit)
            overlap = len(cot_tokens & crit_tokens)
            thresh = 1 if len(crit_tokens) < 8 else 2
            base_sat = overlap >= thresh

            crit_lower = crit.lower()
            is_negative = any(w in crit_lower for w in NEGATIVE_CUE_WORDS)
            if is_negative:
                # for penalty-like rubric, satisfied means flaw is present
                out.append(bool(base_sat))
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
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: List[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = self._client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        obj = r.json()
        return obj["choices"][0]["message"]["content"]


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
            return img_data[0]
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

    def generate(self, question: str, response: str, cot: str, **kwargs) -> RubricProposal:
        image = _extract_first_image_from_kwargs(kwargs)
        image_url = _pil_to_data_url(image)

        trajectory_history = str(kwargs.get("trajectory_history", "") or "None (Initial Turn)")
        turn_idx = kwargs.get("turn_idx", None)
        if turn_idx is None:
            turn_id = "turn_000001"
        else:
            turn_id = f"turn_{int(turn_idx):06d}" if _to_int_or_none(turn_idx) is not None else "turn_000001"

        system_msg = {
            "role": "system",
            "content": _compose_rubricator_system_content(trajectory_history),
        }

        if image_url is not None:
            user_content = [
                {"type": "text", "text": USER_CONTENT},
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": (
                        f"\n[Turn ID]\n{turn_id}\n"
                        f"\n[Question]\n{question or ''}\n"
                        f"\n[Solver Response]\n{response or ''}\n"
                        f"\n[Extracted CoT]\n{cot or ''}\n"
                        f"\n[Constraint]\nKeep at most {self.max_rubrics} rubrics in the output list."
                    ),
                },
            ]
            user_msg = {"role": "user", "content": user_content}
        else:
            user_msg = {
                "role": "user",
                "content": (
                    f"{USER_CONTENT}\n\n"
                    "[No image attached in this call. Use text fallback.]\n"
                    f"[Turn ID]\n{turn_id}\n"
                    f"[Question]\n{question or ''}\n"
                    f"[Solver Response]\n{response or ''}\n"
                    f"[Extracted CoT]\n{cot or ''}\n"
                    f"[Constraint]\nKeep at most {self.max_rubrics} rubrics in the output list."
                ),
            }

        raw = self.client.chat([system_msg, user_msg], temperature=0.0, max_tokens=1500)
        return _parse_rubric_proposal_from_raw(raw=raw, max_rubrics=self.max_rubrics)


class LLMVerifier(BaseVerifier):
    def __init__(self, cfg: dict):
        self.client = OpenAICompatibleClient(
            base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            model=cfg.get("model", ""),
            api_key=cfg.get("api_key"),
            timeout=float(cfg.get("timeout", 60.0)),
        )

    def judge_many(self, criteria: List[str], question: str, response: str, cot: str, **kwargs) -> List[bool]:
        prompt = {
            "role": "user",
            "content": (
                "Evaluate each criterion as strict True/False against the response reasoning.\n"
                "Return JSON only: {\"judgement\": [bool, ...]} with same order.\n"
                f"Question:\n{question}\n\n"
                f"Response:\n{response}\n\n"
                f"Criteria:\n{json.dumps(criteria, ensure_ascii=False)}"
            ),
        }
        raw = self.client.chat([prompt], temperature=0.0, max_tokens=1000)
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
    return json.dumps({"mode": mode, "cfg": cfg}, sort_keys=True, ensure_ascii=False)


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


def _build_verifier(verifier_cfg: dict) -> BaseVerifier:
    mode = str(verifier_cfg.get("mode", "heuristic")).lower()
    key = _make_cache_key(mode, verifier_cfg)
    if key in _VERIFIER_CACHE:
        return _VERIFIER_CACHE[key]

    if mode in {"heuristic", "rule"}:
        inst: BaseVerifier = HeuristicVerifier()
    else:
        inst = LLMVerifier(verifier_cfg)

    _VERIFIER_CACHE[key] = inst
    return inst


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
    return 1.0 if val > 0.0 else -1.0


def _norm_minmax(agg: float, min_v: float, max_v: float) -> float:
    if max_v - min_v <= 1e-12:
        return 0.0
    x = (agg - min_v) / (max_v - min_v)
    return float(max(0.0, min(1.0, x)))


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
    lambda_cot: float = 1.0,
    outcome_weight: float = 1.0,
    rubricator: Optional[dict] = None,
    verifier: Optional[dict] = None,
    fallback_to_heuristic: bool = True,
    **kwargs,
) -> List[dict]:
    alpha = float(alpha)
    lambda_cot = float(lambda_cot)
    outcome_weight = float(outcome_weight)
    rubricator_cfg = rubricator or {}
    verifier_cfg = verifier or {}
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

        try:
            proposal = RubricProposal(rubrics=[], format_ok=False, raw="pending")
        except Exception:
            if not fallback_to_heuristic:
                proposal = RubricProposal(rubrics=[], format_ok=False, raw="rubricator_error")
            else:
                proposal = HeuristicRubricator().generate(question=q, response=sol, cot=cot)

        outcome = _outcome_reward(data_sources[i], sol, ground_truths[i], extra)

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
                "proposal": proposal,
                "outcome": outcome,
                "extra_info": extra,
            }
        )

    # Build trajectory histories first, then call rubricator with history-aware prompts.
    histories = _build_trajectory_history_for_items(items)
    for i in range(n):
        if rubricator_mode == "policy":
            precomputed = _extract_policy_rubric_proposal_from_extra(
                extra_info=items[i].get("extra_info", {}),
                max_rubrics=policy_max_rubrics,
            )
            if precomputed is not None and precomputed.format_ok and len(precomputed.rubrics) > 0:
                items[i]["proposal"] = precomputed
                continue

        try:
            items[i]["proposal"] = rub.generate(
                question=items[i]["question"],
                response=items[i]["solution"],
                cot=items[i]["cot"],
                trajectory_history=histories[i],
                turn_idx=items[i].get("turn_idx", None),
                extra_info=items[i].get("extra_info", {}),
            )
        except Exception:
            if not fallback_to_heuristic:
                items[i]["proposal"] = RubricProposal(rubrics=[], format_ok=False, raw="rubricator_error")
            else:
                items[i]["proposal"] = HeuristicRubricator().generate(
                    question=items[i]["question"],
                    response=items[i]["solution"],
                    cot=items[i]["cot"],
                )

    group_to_indices: Dict[str, List[int]] = {}
    for i, x in enumerate(items):
        group_to_indices.setdefault(x["group"], []).append(i)

    cot_rewards = [0.0] * n
    valid_ratios = [0.0] * n
    evolving_rewards = [0.0] * n
    corr_means = [0.0] * n

    for _, idxs in group_to_indices.items():
        z = np.array([1 if items[t]["outcome"] > 0 else 0 for t in idxs], dtype=float)
        group_corrs: List[float] = []

        for i in idxs:
            proposal: RubricProposal = items[i]["proposal"]
            rubrics = proposal.rubrics
            if len(rubrics) == 0:
                cot_rewards[i] = 0.0
                valid_ratios[i] = 0.0
                evolving_rewards[i] = 1.0 if proposal.format_ok else 0.0
                continue

            valid_flags: List[bool] = []
            sat_self: List[bool] = []
            points: List[float] = []
            local_corrs: List[float] = []

            criteria = [r.criterion for r in rubrics]
            points = [float(r.points) for r in rubrics]

            # evaluate each criterion over all trajectories in same question-group
            v_matrix = np.zeros((len(criteria), len(idxs)), dtype=float)
            for j_pos, j in enumerate(idxs):
                try:
                    judged = ver.judge_many(
                        criteria=criteria,
                        question=items[j]["question"],
                        response=items[j]["solution"],
                        cot=items[j]["cot"],
                    )
                except Exception:
                    if fallback_to_heuristic:
                        judged = HeuristicVerifier().judge_many(
                            criteria=criteria,
                            question=items[j]["question"],
                            response=items[j]["solution"],
                            cot=items[j]["cot"],
                        )
                    else:
                        judged = [False] * len(criteria)
                judged = [bool(x) for x in judged[: len(criteria)]]
                if len(judged) < len(criteria):
                    judged.extend([False] * (len(criteria) - len(judged)))
                v_matrix[:, j_pos] = np.array([1.0 if b else 0.0 for b in judged], dtype=float)

            i_pos = idxs.index(i)
            sat_self = [bool(v_matrix[k, i_pos] > 0.5) for k in range(len(criteria))]

            for k in range(len(criteria)):
                vk = v_matrix[k, :]
                corr = _corr_binary(vk, z)
                local_corrs.append(corr)
                is_valid = bool(corr > alpha and float(np.std(vk)) > 0.0)
                valid_flags.append(is_valid)
                if is_valid:
                    group_corrs.append(corr)

            valid_idx = [k for k, flag in enumerate(valid_flags) if flag]
            k_total = len(criteria)
            k_valid = len(valid_idx)
            valid_ratio = (k_valid / k_total) if k_total > 0 else 0.0
            valid_ratios[i] = float(valid_ratio)

            format_reward = 1.0 if proposal.format_ok else 0.0
            evolving_rewards[i] = float(valid_ratio + format_reward)

            if k_valid == 0:
                cot_rewards[i] = 0.0
                continue

            agg = 0.0
            min_v = 0.0
            max_v = 0.0
            for k in valid_idx:
                p = points[k]
                sat = sat_self[k]
                if sat:
                    agg += p
                max_v += max(0.0, p)
                min_v += min(0.0, p)

            cot_rewards[i] = _norm_minmax(agg, min_v, max_v)

        group_corr_mean = float(np.mean(group_corrs)) if group_corrs else 0.0
        for i in idxs:
            corr_means[i] = group_corr_mean

    outputs: List[dict] = []
    for i in range(n):
        outcome = float(items[i]["outcome"])
        r_cot = float(cot_rewards[i])
        score = float(outcome_weight * outcome + lambda_cot * r_cot)
        outputs.append(
            {
                "score": score,
                "outcome_reward": outcome,
                "cot_reward": r_cot,
                "rubricator_reward": float(evolving_rewards[i]),
                "valid_ratio": float(valid_ratios[i]),
                "corr_mean": float(corr_means[i]),
            }
        )
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
