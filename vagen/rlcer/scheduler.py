"""Rubricator scheduler for anti-Nash equilibrium and plateau detection.

Monitors training signals (corr_mean, valid_ratio, trivial_scale, validation
metrics) and dynamically adjusts RLCER control knobs (lambda_cot,
rubricator_loss_weight, pure_outcome, enable_dual_role_update) to prevent
the rubricator-reasoner system from collapsing into a Nash equilibrium where
both achieve high internal rewards without improving actual task outcome.

Four mechanisms:
1. Warmup: Linearly ramp lambda_cot and rubricator_loss_weight from 0 to
   their configured targets over the first N steps.
2. Correlation-based gating: Decay weights when rubric-outcome correlation
   (corr_mean) drops below a threshold, and permanently freeze the
   rubricator if weights hit their minimums.
3. Plateau detection: Freeze the rubricator if validation
   performance stagnates for N consecutive evaluations.
4. Role-health recovery: suspend unreliable rubric RL when proposal formatting
   collapses, then warm-restart it after sustained recovery.

All freeze decisions are permanent for the rest of the training run.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf, open_dict


class RubricatorScheduler:
    """Runtime scheduler that adjusts RLCER control knobs based on training signals.

    Config is read from ``custom_reward_function.reward_kwargs.scheduler``.
    When all sub-mechanisms are disabled (the default), the scheduler is a
    no-op and behaviour is identical to not having a scheduler at all.
    """

    def __init__(self, config: Any) -> None:
        reward_kwargs = (
            config.get("custom_reward_function", {})
            .get("reward_kwargs", {})
        )
        cfg = reward_kwargs.get("scheduler", {})

        # ---- Plateau detection ----
        self.plateau_enabled: bool = bool(cfg.get("plateau_enabled", False))
        self.plateau_patience: int = int(cfg.get("plateau_patience", 5))
        self.plateau_rel_threshold: float = float(
            cfg.get("plateau_rel_threshold", 0.05)
        )
        self.plateau_metric_key: str = str(
            cfg.get("plateau_metric_key", "val-core/sokoban/reward/mean@8")
        )

        # ---- Correlation-based gating ----
        self.corr_gating_enabled: bool = bool(cfg.get("corr_gating_enabled", False))
        self.corr_threshold: float = float(cfg.get("corr_threshold", 0.1))
        self.corr_patience: int = int(cfg.get("corr_patience", 10))
        self.corr_weight_decay: float = float(cfg.get("corr_weight_decay", 0.5))

        # ---- Warmup ----
        self.warmup_enabled: bool = bool(cfg.get("warmup_enabled", False))
        self.warmup_steps: int = int(cfg.get("warmup_steps", 100))
        self.warmup_from_lambda_cot: float = float(
            cfg.get("warmup_from_lambda_cot", 0.0)
        )
        self.warmup_from_rub_loss_weight: float = float(
            cfg.get("warmup_from_rub_loss_weight", 0.0)
        )

        # ---- Annealing minimums ----
        self.anneal_min_lambda_cot: float = float(
            cfg.get("anneal_min_lambda_cot", 0.0)
        )
        self.anneal_min_rub_loss_weight: float = float(
            cfg.get("anneal_min_rub_loss_weight", 0.0)
        )

        # ---- Role-health recovery ----
        self.recovery_enabled: bool = bool(cfg.get("recovery_enabled", False))
        self.recovery_enter_threshold: float = float(
            cfg.get("recovery_enter_threshold", 0.80)
        )
        self.recovery_exit_threshold: float = float(
            cfg.get("recovery_exit_threshold", 0.95)
        )
        self.recovery_enter_patience: int = max(
            1, int(cfg.get("recovery_enter_patience", 5))
        )
        self.recovery_exit_patience: int = max(
            1, int(cfg.get("recovery_exit_patience", 20))
        )
        self.recovery_warm_restart_steps: int = max(
            1, int(cfg.get("recovery_warm_restart_steps", 50))
        )

        # ---- Store original config values (served as warmup targets) ----
        self._original_lambda_cot: float = float(
            reward_kwargs.get("lambda_cot", 1.0)
        )
        self._original_rub_loss_weight: float = float(
            reward_kwargs.get("rubricator_loss_weight", 0.5)
        )
        self._original_pure_outcome: bool = bool(
            reward_kwargs.get("pure_outcome", False)
        )
        self._original_enable_dual_role: bool = bool(
            reward_kwargs.get("enable_dual_role_update", False)
        )

        # ---- Internal state ----
        self._step: int = 0
        self._frozen: bool = False  # permanent freeze flag
        self._freeze_reason: str = ""  # "plateau" or "corr_gating"

        # Plateau detection state
        self._plateau_history: List[float] = []
        self._plateau_counter: int = 0

        # Correlation gating state
        self._corr_ema: float = 1.0
        self._corr_ema_beta: float = 0.9
        self._corr_low_counter: int = 0
        self._valid_ratio_ema: float = 1.0
        self._trivial_scale_ema: float = 1.0
        self._corr_gating_started: bool = False

        # Role-health state: normal -> recovery -> warm_restart -> normal.
        self._health_state: str = "normal"
        self._health_bad_steps: int = 0
        self._health_good_steps: int = 0
        self._warm_restart_start_step: int = -1
        self._proposal_format_ok_ema: float = 1.0
        self._proposal_nonempty_ema: float = 1.0

        # Current effective values (written to config each step)
        self._curr_lambda_cot: float = self._original_lambda_cot
        self._curr_rub_loss_weight: float = self._original_rub_loss_weight
        self._curr_pure_outcome: bool = self._original_pure_outcome
        self._curr_enable_dual_role: bool = self._original_enable_dual_role

        # Quick check: if user manually set pure_outcome=True, respect it
        if self._original_pure_outcome:
            self._frozen = True
            self._freeze_reason = "manual"

        # Whether any scheduler mechanism is active
        self._any_active: bool = (
            self.warmup_enabled
            or self.corr_gating_enabled
            or self.plateau_enabled
            or self.recovery_enabled
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        config: Any,
        global_steps: int,
        reward_extra_infos_dict: Dict[str, list],
        metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called once per training step after reward computation.

        Reads training signals, applies scheduling logic, and mutates
        ``config`` in-place so that downstream code sees the updated values.

        Args:
            config: The OmegaConf config object (mutated in-place).
            global_steps: Current training step number.
            reward_extra_infos_dict: Per-sample reward metadata from
                ``_compute_score_batched`` (keys include ``corr_mean``,
                ``valid_ratio``, ``trivial_scale``, etc.).
            metrics: Training metrics dict (scheduler will append its own
                metrics under ``rlcer/scheduler/``).
            val_metrics: Validation metrics dict from ``_validate()``.
                Only provided on steps where validation runs.
        """
        self._step = global_steps

        if not self._any_active:
            return

        if self._frozen:
            self._log_metrics(metrics)
            self._write_config(config)
            return

        # 1. Warmup
        warmup_progress = self._apply_warmup()

        # 2. Correlation gating (only after warmup completes)
        if not warmup_progress < 1.0:
            self._apply_corr_gating(reward_extra_infos_dict)

        # 3. Plateau detection (only after warmup completes and when val_metrics available)
        if val_metrics is not None and warmup_progress >= 1.0:
            self._apply_plateau_detection(val_metrics)

        # 4. Check freeze condition (skip during warmup)
        if warmup_progress >= 1.0:
            self._check_freeze()

        # 4b. Role-health recovery is applied last so correlation/plateau
        # scheduling cannot accidentally re-enable an unhealthy rubric update.
        # Keep pure_outcome=False: generation must continue for health probes.
        if self.recovery_enabled and not self._frozen:
            self._apply_role_health_recovery(reward_extra_infos_dict)

        # 5. Write effective values to config
        self._write_config(config)

        # 6. Log metrics
        self._log_metrics(metrics)

    def _apply_role_health_recovery(
        self, reward_extra_infos_dict: Dict[str, list]
    ) -> None:
        """Suspend and later warm-restart rubric RL after generation collapse."""
        fmt_values = reward_extra_infos_dict.get("proposal_format_ok", [])
        nonempty_values = reward_extra_infos_dict.get("proposal_nonempty", [])
        if not fmt_values or not nonempty_values:
            return

        fmt = float(np.mean(fmt_values))
        nonempty = float(np.mean(nonempty_values))
        self._proposal_format_ok_ema = 0.9 * self._proposal_format_ok_ema + 0.1 * fmt
        self._proposal_nonempty_ema = 0.9 * self._proposal_nonempty_ema + 0.1 * nonempty
        health = min(fmt, nonempty)

        if self._health_state == "normal":
            if health < self.recovery_enter_threshold:
                self._health_bad_steps += 1
            else:
                self._health_bad_steps = 0
            if self._health_bad_steps >= self.recovery_enter_patience:
                self._health_state = "recovery"
                self._health_good_steps = 0
                self._curr_lambda_cot = 0.0
                self._curr_rub_loss_weight = 0.0
                self._curr_enable_dual_role = False
        elif self._health_state == "recovery":
            self._curr_lambda_cot = 0.0
            self._curr_rub_loss_weight = 0.0
            self._curr_enable_dual_role = False
            if health >= self.recovery_exit_threshold:
                self._health_good_steps += 1
            else:
                self._health_good_steps = 0
            if self._health_good_steps >= self.recovery_exit_patience:
                self._health_state = "warm_restart"
                self._warm_restart_start_step = self._step
        elif self._health_state == "warm_restart":
            progress = min(
                1.0,
                (self._step - self._warm_restart_start_step)
                / self.recovery_warm_restart_steps,
            )
            self._curr_lambda_cot = self._original_lambda_cot * progress
            self._curr_rub_loss_weight = self._original_rub_loss_weight * progress
            self._curr_enable_dual_role = self._original_enable_dual_role
            if health < self.recovery_enter_threshold:
                self._health_state = "recovery"
                self._health_good_steps = 0
            elif progress >= 1.0:
                self._health_state = "normal"
                self._health_bad_steps = 0

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _apply_warmup(self) -> float:
        """Linear warmup of lambda_cot and rubricator_loss_weight.

        Returns progress in [0.0, 1.0].  1.0 means warmup is complete.
        """
        if not self.warmup_enabled:
            return 1.0

        if self._step > self.warmup_steps:
            return 1.0

        progress = self._step / max(self.warmup_steps, 1)
        progress = min(progress, 1.0)

        self._curr_lambda_cot = (
            self.warmup_from_lambda_cot
            + progress * (self._original_lambda_cot - self.warmup_from_lambda_cot)
        )
        self._curr_rub_loss_weight = (
            self.warmup_from_rub_loss_weight
            + progress
            * (self._original_rub_loss_weight - self.warmup_from_rub_loss_weight)
        )
        return progress

    # ------------------------------------------------------------------
    # Correlation-based gating
    # ------------------------------------------------------------------

    def _apply_corr_gating(
        self, reward_extra_infos_dict: Dict[str, list]
    ) -> None:
        """Decay weights when rubric-outcome correlation drops below threshold."""
        if not self.corr_gating_enabled:
            return

        if self.warmup_enabled and self._step <= self.warmup_steps:
            return

        if not self._corr_gating_started:
            self._corr_gating_started = True
            self._corr_ema = 1.0
            self._corr_low_counter = 0

        corr_values = reward_extra_infos_dict.get("corr_mean", [])
        vr_values = reward_extra_infos_dict.get("valid_ratio", [])
        ts_values = reward_extra_infos_dict.get("trivial_scale", [])

        if len(corr_values) == 0:
            return

        step_corr = float(np.mean(corr_values))

        # corr_mean=0 means correlation was not computable (e.g. n=1 rollout
        # or zero outcome variance in the group).  Treat as "no signal" and
        # skip the EMA update so it doesn't drag corr_ema toward 0.
        if step_corr == 0.0:
            return

        self._corr_ema = (
            self._corr_ema_beta * self._corr_ema
            + (1 - self._corr_ema_beta) * step_corr
        )
        if len(vr_values) > 0:
            step_vr = float(np.mean(vr_values))
            self._valid_ratio_ema = (
                0.9 * self._valid_ratio_ema + 0.1 * step_vr
            )
        if len(ts_values) > 0:
            step_ts = float(np.mean(ts_values))
            self._trivial_scale_ema = (
                0.9 * self._trivial_scale_ema + 0.1 * step_ts
            )

        if self._corr_ema < self.corr_threshold:
            self._corr_low_counter += 1
        else:
            self._corr_low_counter = max(0, self._corr_low_counter - 1)

        n_windows = self._corr_low_counter // self.corr_patience
        decay = self.corr_weight_decay ** n_windows

        self._curr_lambda_cot = max(
            self._original_lambda_cot * decay,
            self.anneal_min_lambda_cot,
        )
        self._curr_rub_loss_weight = max(
            self._original_rub_loss_weight * decay,
            self.anneal_min_rub_loss_weight,
        )

    # ------------------------------------------------------------------
    # Plateau detection
    # ------------------------------------------------------------------

    def _apply_plateau_detection(self, val_metrics: Dict[str, float]) -> None:
        """Freeze rubricator if validation performance stagnates."""
        if not self.plateau_enabled:
            return

        val = val_metrics.get(self.plateau_metric_key)
        if val is None:
            if not hasattr(self, "_plateau_key_warned"):
                warnings.warn(
                    f"[RubricatorScheduler] plateau_metric_key "
                    f"'{self.plateau_metric_key}' not found in val_metrics. "
                    f"Available keys: {list(val_metrics.keys())[:10]}...",
                    stacklevel=2,
                )
                self._plateau_key_warned = True
            return

        self._plateau_history.append(float(val))

        if len(self._plateau_history) < self.plateau_patience + 1:
            return

        recent = self._plateau_history[-1]
        reference = self._plateau_history[-(self.plateau_patience + 1)]
        rel_change = abs(recent - reference) / max(abs(reference), 1e-8)

        if rel_change < self.plateau_rel_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        if self._plateau_counter >= 3:
            self._frozen = True
            self._freeze_reason = "plateau"
            print(
                f"[RubricatorScheduler] PLATEAU DETECTED at step {self._step}: "
                f"'{self.plateau_metric_key}' stagnant for {self.plateau_patience} "
                f"validations (rel_change={rel_change:.4f} < "
                f"{self.plateau_rel_threshold:.4f}). "
                f"Freezing rubricator permanently."
            )

    # ------------------------------------------------------------------
    # Freeze check
    # ------------------------------------------------------------------

    def _check_freeze(self) -> None:
        """Check if correlation gating should trigger a permanent freeze."""
        if self._frozen:
            return

        if not self.corr_gating_enabled:
            return

        min_counter_for_freeze = self.corr_patience * 3
        _eps = 1e-4
        if (
            self._corr_low_counter >= min_counter_for_freeze
            and self._curr_lambda_cot < self.anneal_min_lambda_cot + _eps
            and self._curr_rub_loss_weight < self.anneal_min_rub_loss_weight + _eps
        ):
            self._frozen = True
            self._freeze_reason = "corr_gating"
            print(
                f"[RubricatorScheduler] CORR GATING FREEZE at step {self._step}: "
                f"corr_ema={self._corr_ema:.4f} < threshold={self.corr_threshold:.4f} "
                f"for {self._corr_low_counter} steps (min={min_counter_for_freeze}). "
                f"Both weights hit minimums. "
                f"Freezing rubricator permanently."
            )

    # ------------------------------------------------------------------
    # Write config
    # ------------------------------------------------------------------

    def _write_config(self, config: Any) -> None:
        """Mutate the live config so downstream code sees updated knob values."""
        if self._frozen:
            self._curr_pure_outcome = True
            self._curr_enable_dual_role = False
            self._curr_lambda_cot = self.anneal_min_lambda_cot
            self._curr_rub_loss_weight = self.anneal_min_rub_loss_weight

        if self._original_pure_outcome:
            self._curr_pure_outcome = True
        if not self._original_enable_dual_role:
            self._curr_enable_dual_role = False

        try:
            with open_dict(config):
                rk = config.custom_reward_function.reward_kwargs
                rk.lambda_cot = self._curr_lambda_cot
                rk.rubricator_loss_weight = self._curr_rub_loss_weight
                rk.pure_outcome = self._curr_pure_outcome
                rk.enable_dual_role_update = self._curr_enable_dual_role
        except Exception:
            reward_kwargs = config.get("custom_reward_function", {}).get(
                "reward_kwargs", {}
            )
            if hasattr(reward_kwargs, "__setitem__") or hasattr(
                reward_kwargs, "__setattr__"
            ):
                try:
                    reward_kwargs["lambda_cot"] = self._curr_lambda_cot
                    reward_kwargs["rubricator_loss_weight"] = (
                        self._curr_rub_loss_weight
                    )
                    reward_kwargs["pure_outcome"] = self._curr_pure_outcome
                    reward_kwargs["enable_dual_role_update"] = (
                        self._curr_enable_dual_role
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Append scheduler state to the training metrics dict for wandb."""
        metrics["rlcer/scheduler/corr_ema"] = self._corr_ema
        metrics["rlcer/scheduler/corr_low_counter"] = self._corr_low_counter
        metrics["rlcer/scheduler/valid_ratio_ema"] = self._valid_ratio_ema
        metrics["rlcer/scheduler/trivial_scale_ema"] = self._trivial_scale_ema
        metrics["rlcer/scheduler/lambda_cot_effective"] = self._curr_lambda_cot
        metrics["rlcer/scheduler/rub_loss_weight_effective"] = (
            self._curr_rub_loss_weight
        )
        metrics["rlcer/scheduler/pure_outcome_active"] = float(
            self._curr_pure_outcome
        )
        metrics["rlcer/scheduler/plateau_frozen"] = float(
            self._frozen and self._freeze_reason == "plateau"
        )
        metrics["rlcer/scheduler/corr_gate_frozen"] = float(
            self._frozen and self._freeze_reason == "corr_gating"
        )
        state_id = {"normal": 0.0, "recovery": 1.0, "warm_restart": 2.0}
        metrics["rlcer/scheduler/health_state"] = state_id[self._health_state]
        metrics["rlcer/scheduler/recovery_active"] = float(
            self._health_state == "recovery"
        )
        metrics["rlcer/scheduler/health_bad_steps"] = self._health_bad_steps
        metrics["rlcer/scheduler/health_good_steps"] = self._health_good_steps
        metrics["rlcer/scheduler/proposal_format_ok_ema"] = (
            self._proposal_format_ok_ema
        )
        metrics["rlcer/scheduler/proposal_nonempty_ema"] = (
            self._proposal_nonempty_ema
        )

        if self.warmup_enabled and self._step <= self.warmup_steps:
            warmup_progress = self._step / max(self.warmup_steps, 1)
            metrics["rlcer/scheduler/warmup_active"] = 1.0
            metrics["rlcer/scheduler/warmup_progress"] = warmup_progress
        else:
            metrics["rlcer/scheduler/warmup_active"] = 0.0
            metrics["rlcer/scheduler/warmup_progress"] = -1.0

        if self.plateau_enabled:
            metrics["rlcer/scheduler/plateau_history_len"] = len(
                self._plateau_history
            )
            if self._plateau_history:
                metrics["rlcer/scheduler/last_val_metric"] = (
                    self._plateau_history[-1]
                )
            metrics["rlcer/scheduler/plateau_counter"] = self._plateau_counter
