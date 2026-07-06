#!/usr/bin/env python3
"""
Trinity agent-runtime entry point — the in-Job orchestrator.

Runs inside the detached EKS Job (per managed-agent-runner.md). It:
  1. installs the Bedrock Converse dispatch + pricing monkeypatches over the
     vendored fugu package (the only adaptation; algorithm/loop stay upstream),
  2. rewrites fugu's worker registry so every worker is a Bedrock call with NO
     server/port (so fugu never takes the vLLM-HTTP path),
  3. runs the requested phase (eval | smoke | train) against the bundled split,
  4. streams checkpoints + rollouts + es_log to S3 every CMA-ES iteration
     (always including iter 0 — the cost-aware-routing reclaim lesson),
  5. enforces a hard Bedrock cost cap and the Phase-0.5 throttle/non-degeneracy
     gates, halting before the expensive phase if any gate fails.

Credentials: IRSA (SigV4) for the 6 Converse workers + S3; the optional GPT-5.5
ord-0 swap additionally reads BEDROCK_BEARER_TOKEN from a mounted K8s secret.

This file does NOT create AWS infra — terraform/ provisions the IRSA role + bucket
(Gate 0.0 step 0). It assumes both exist.

Usage (inside the Job):
  python run_trinity_agent.py --phase eval  \
      --vendor-root vendor/trinity-upstream \
      --s3-uri s3://<bench-bucket>/trinity-coordinator/<run-id>/ \
      --model-file vendor/trinity-upstream/logs/ckpt/models/model_iter_60.npy
  python run_trinity_agent.py --phase smoke --iters 3 --cost-cap-usd 250
  python run_trinity_agent.py --phase train --iters 60 --cost-cap-usd 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# scripts/ dir on path so the sibling adaptation modules import cleanly.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


def _log(msg: str) -> None:
    print(f"[trinity-agent] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Monkeypatch install — must run BEFORE fugu worker calls, AFTER fugu import.
# ---------------------------------------------------------------------------
def install_bedrock_adaptation(vendor_root: Path) -> None:
    sys.path.insert(0, str(vendor_root))  # make `fugu` importable

    # CRITICAL (live-verified 2026-06-24): fugu's JobManager spawns Pool workers via
    # mp.set_start_method("spawn"); those workers re-import fugu.utils FRESH and do NOT
    # inherit a main-process-only monkeypatch — they fall back to the original
    # OpenAI/Together/Gemini clients, fail (CLOSE-WAIT), and stall the run. We export
    # the env that scripts/sitecustomize.py reads, so EVERY spawned interpreter
    # re-installs the Bedrock dispatch at startup. (sitecustomize is on PYTHONPATH
    # because this scripts dir is.) See lessons.md finding #8.
    os.environ["CAR_TRINITY_BEDROCK_PATCH"] = "1"
    os.environ["CAR_TRINITY_VENDOR_ROOT"] = str(vendor_root)
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = scripts_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Spend/throttle sink dir — inherited by spawned workers via env; each worker
    # flushes its cumulative cost+throttle here, main aggregates (lessons #20).
    # Start clean so a re-run doesn't sum stale PID files from a prior run.
    tdir = os.environ.setdefault(
        "CAR_TRINITY_TELEMETRY_DIR", str(vendor_root / "logs" / "run" / "_telemetry"))
    try:
        import glob as _glob
        os.makedirs(tdir, exist_ok=True)
        for _f in _glob.glob(os.path.join(tdir, "cost_*.json")) + \
                  _glob.glob(os.path.join(tdir, "throttle_*.json")):
            os.remove(_f)
    except Exception:
        pass

    import fugu.utils  # noqa: F401  (ensure module objects exist to rebind)
    import fugu.cost   # noqa: F401

    import bedrock_clients
    import cost_bedrock
    import routing_policy

    bedrock_clients.install()
    cost_bedrock.install()
    routing_policy.install()   # baseline routing override (no-op unless CAR_TRINITY_ROUTING_POLICY set)
    _log("installed Bedrock Converse dispatch + pricing over fugu (main + spawn-worker via sitecustomize)")


def bedrock_agent_configs() -> dict:
    """fugu agent registry with Bedrock workers and NO ports (forces cloud path)."""
    from worker_pool_bedrock import AGENT_CONFIGS, LLM_NAMES
    # Strip any port so fugu.utils._resolve_agent_complete_info returns port=None
    # -> server=None -> the (patched) cloud-API branch, never vLLM HTTP.
    cfgs = {}
    for name, c in AGENT_CONFIGS.items():
        cc = dict(c)
        cc.pop("port", None)
        cfgs[name] = cc
    return cfgs, LLM_NAMES


# ---------------------------------------------------------------------------
# S3 sync — cheap state, sync often.
# ---------------------------------------------------------------------------
def s3_sync(local_dir: Path, s3_uri: str, *, delete: bool = False) -> None:
    if not s3_uri:
        return
    cmd = ["aws", "s3", "sync", str(local_dir), s3_uri.rstrip("/") + "/"]
    if delete:
        cmd.append("--delete")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        _log(f"synced {local_dir} -> {s3_uri}")
    except subprocess.CalledProcessError as e:
        _log(f"WARNING s3 sync failed: {e.stderr[:300]}")
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING s3 sync error: {e}")


def s3_put_json(obj: dict, s3_uri: str, key: str) -> None:
    if not s3_uri:
        return
    tmp = SCRIPTS_DIR / f".{key.replace('/', '_')}"
    tmp.write_text(json.dumps(obj, indent=2))
    dest = s3_uri.rstrip("/") + "/" + key
    try:
        subprocess.run(["aws", "s3", "cp", str(tmp), dest], check=True,
                       capture_output=True, text=True, timeout=120)
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING s3 cp {key} failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Pre-run snapshots & gate helpers
# ---------------------------------------------------------------------------
def snapshot_prices(s3_uri: str) -> None:
    import cost_bedrock
    from worker_pool_bedrock import POOL
    snap = {w.friendly_name: [cost_bedrock.PRICES[w.friendly_name].in_per_1m,
                              cost_bedrock.PRICES[w.friendly_name].out_per_1m]
            for w in POOL}
    snap["_version"] = cost_bedrock._VERSION
    s3_put_json(snap, s3_uri, "verified_prices_snapshot.json")
    _log(f"price snapshot pinned ({cost_bedrock._VERSION})")


def worker_entropy(agent_distribution: dict) -> float:
    counts = [c for c in agent_distribution.values() if c is not None]
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p)  # nats
    return ent


def check_phase05_gates(stats: dict) -> tuple[bool, list[str]]:
    """Phase 0.5 exit criteria — all must pass before the full run."""
    failures: list[str] = []
    dist = stats.get("agent_distribution", {})
    roles = stats.get("role_usage", {})  # {"solver":n,"thinker":n,"verifier":n}
    tele = stats.get("throttle", {})

    # 1. Worker non-degeneracy — every worker selected ≥once.
    unused = [name for name, c in dist.items() if not c]
    if unused:
        failures.append(f"worker degeneracy: unused workers {unused}")

    # 2. Role non-degeneracy — all three roles used + ≥1 verifier early halt.
    for r in ("solver", "thinker", "verifier"):
        if not roles.get(r):
            failures.append(f"role degeneracy: '{r}' never used")
    if not stats.get("verifier_early_halts"):
        failures.append("verifier triggered no early halts")

    # 3. Throttle survival — zero dropped episodes at the 512-candidate burst.
    if tele.get("dropped", 0) > 0:
        failures.append(f"throttle: {tele['dropped']} dropped episodes "
                        f"(rate {tele.get('dropped_rate', 0):.3%})")

    # 4. Question-conditioning — entropy > 1.5 nats.
    ent = worker_entropy(dist)
    if ent <= 1.5:
        failures.append(f"entropy {ent:.3f} nats <= 1.5 (head not conditioning)")

    # 5. Brand/concentration bias — no single worker > 40% of selections.
    # Entropy (above) measures dispersion symmetrically but can pass while one
    # worker still dominates; rl-conductor found the base model had a ~52% Opus
    # brand-skew that biased the reward signal (grpo-router-negative-result.md,
    # MEMORY brand-bias). 40% (vs uniform ~14% over 7 workers) flags over-
    # concentration that entropy alone may miss.
    total = sum(c for c in dist.values() if c) or 0
    if total:
        top_name, top_c = max(dist.items(), key=lambda kv: (kv[1] or 0))
        frac = (top_c or 0) / total
        if frac > 0.40:
            failures.append(f"brand/concentration bias: '{top_name}' = {frac:.0%} "
                            f"of selections (> 40%); head may be position/brand-skewed "
                            f"like rl-conductor's ~52% Opus skew")

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def run_eval(args) -> int:
    """Phase 0 — eval a trained checkpoint on the bundled LiveCodeBench split."""
    vendor = Path(args.vendor_root).resolve()
    install_bedrock_adaptation(vendor)
    cfgs, llm_names = bedrock_agent_configs()

    model_file = Path(args.model_file) if args.model_file else None
    if model_file is None or not model_file.exists():
        _log(f"FATAL: checkpoint not found: {model_file}. The bundled "
             f"model_iter_60.npy is ABSENT from the vendored copy (only es_log.json "
             f"+ data split are present). Phase 0 cannot run without it — obtain the "
             f"checkpoint from the OpenReview supplementary or train one via --phase train.")
        return 2

    # The vendored evaluate_trinity_livecodebench.py reads agent configs from its
    # own AVAILABLE_OPEN_AGENTS map; override it to our Bedrock pool (no ports),
    # then delegate. We import it after the monkeypatch so its fugu calls route to
    # Bedrock.
    import evaluate_trinity_livecodebench as ev
    import json as _json

    # The trained head is POSITIONAL: output dim = ord 0..6. The checkpoint's
    # es_log.json llm_names give that order (gpt-5, claude-sonnet, gemini-pro,
    # deepseek-r1, gemma, qwen-reason, qwen-direct). We must map each of THOSE
    # names — the ones the eval's build_eval_config will look up — to our Bedrock
    # worker by INDEX, so ord i routes to Bedrock pool[i]. Keying by our own
    # friendly-names (previous bug) left the checkpoint names unresolved → the eval
    # fell back to a closed `gpt-4.1` OpenAI call → "missing credentials" on every
    # episode. (lessons.md finding #10.)
    es_log = _json.loads((vendor / "logs" / "ckpt" / "es_log.json").read_text())
    ckpt_names = es_log[0]["configs"]["llm_names"]
    if len(ckpt_names) != len(llm_names):
        _log(f"FATAL: checkpoint has {len(ckpt_names)} workers but Bedrock pool has "
             f"{len(llm_names)} — head output dim mismatch, cannot map positionally.")
        return 2
    # Build AVAILABLE_OPEN_AGENTS keyed by the CHECKPOINT names → our Bedrock cfg by index.
    # The vendored eval's build_evaluation_config assumes every open agent has a
    # `port` (it builds a vLLM server/port map). Our Bedrock dispatch ignores
    # server/port (routes to Converse), but the config builder still reads cfg["port"]
    # → KeyError. Inject a harmless dummy port so the builder is satisfied; the
    # monkeypatched query_locally_hosted_model never uses it. (lessons.md finding #11.)
    # CRITICAL (lessons #12): the rollout env resolves the head's output index via
    # `core.py: agent_name = self.llm_names[agent_id]`, and self.llm_names comes from
    # the eval's selected_agents (= the checkpoint's es_log llm_names, e.g. 'gpt-4.1').
    # If we leave those names, the head index → 'gpt-4.1' → fugu routes by model-name
    # substring to the OpenAI client → no creds → every episode fails (test_score 0.0,
    # agent_distribution shows gpt-4.1:5). The positional-config remap alone is NOT
    # enough — the NAMES the env carries must BE our Bedrock friendly-names so
    # query_bedrock_dispatch resolves them via by_friendly_name.
    #
    # Fix: force the env's llm_names to OUR Bedrock pool order (LLM_NAMES) by
    # monkeypatching parse_selected_agents. The head is positional, so ord i → our
    # worker i. AVAILABLE_OPEN_AGENTS is then keyed by our names (with dummy port).
    remapped = {}
    for i in range(len(llm_names)):
        c = dict(cfgs[llm_names[i]])
        c.setdefault("port", 0)            # dummy; dispatch routes to Bedrock regardless
        c.setdefault("server", "bedrock")  # dummy; satisfies server_map construction
        remapped[llm_names[i]] = c
    ev.AVAILABLE_OPEN_AGENTS = remapped
    ev.CLOSED_LLM_NAMES = []          # every worker is "open" (Bedrock) now
    ev.TOGETHER_FLAGS = {}
    ev.DEFAULT_OPEN_SERVERS = ""      # no vLLM servers

    # Force selected_agents → our Bedrock names (positional). This sets core.py's
    # self.llm_names so head index i resolves to our worker i.
    _orig_parse = ev.parse_selected_agents
    def _patched_parse(config, *a, **kw):
        _sel, mtypes = _orig_parse(config, *a, **kw)
        return list(llm_names), mtypes
    ev.parse_selected_agents = _patched_parse
    _log(f"eval: env llm_names forced to Bedrock pool order: {llm_names}")
    for i, cn in enumerate(ckpt_names):
        _log(f"  ord {i}: {cn}  →  {cfgs[llm_names[i]]['model_name']}")
    # The harness owns argv parsing; build a minimal argv for it.
    sys.argv = ["evaluate_trinity_livecodebench.py", str(vendor / "logs" / "ckpt"),
                "--model-file", str(model_file), "--test-size", str(args.test_size)]
    if args.debug:
        sys.argv.append("--debug")
    rc = ev.main() if hasattr(ev, "main") else 0
    snapshot_prices(args.s3_uri)
    s3_sync(vendor / "logs" / "ckpt", args.s3_uri)
    return rc or 0


def run_training(args, *, smoke: bool) -> int:
    """Phase 0.5 (smoke) / Phase 1 (full) — CMA-ES from scratch on the Bedrock pool."""
    import cost_bedrock
    import bedrock_clients

    vendor = Path(args.vendor_root).resolve()
    install_bedrock_adaptation(vendor)
    cfgs, llm_names = bedrock_agent_configs()
    snapshot_prices(args.s3_uri)

    from fugu.trainer import RouterInfrastructure
    # The vendored CMAEvolutionTrainer is EVAL-ONLY (no train loop — the OpenReview
    # submission omitted it). CMATrainingLoop subclasses it to add the missing
    # cma.ask/tell optimization using the same scoring primitive (submit_training_job)
    # that run_test uses. See scripts/cma_train.py + lessons.md finding #13.
    from cma_train import CMATrainingLoop

    log_dir = Path(args.log_dir or (vendor / "logs" / "run")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    iters = args.iters if args.iters else (3 if smoke else 60)
    # Smoke validates loop MECHANICS (ask/tell, gate-key emission, es_log write,
    # S3 sync, cost-cap, throttle survival) on a small episode budget — NOT a
    # trained model. Full run uses the paper's λ=32, mCMA=16. λ=6×mCMA=3×3 iters
    # ≈ 54 episodes total keeps the smoke cheap (<$30) while exercising every path.
    popsize = 6 if smoke else 32
    num_repeats = 3 if smoke else 16
    _log(f"{'SMOKE' if smoke else 'FULL'} CMA-ES: iters={iters} lambda={popsize} "
         f"mCMA={num_repeats} workers={len(llm_names)} cost_cap=${args.cost_cap_usd}")

    # Single-GPU box (g6e.Nxlarge = 1× L40S → cuda:0 only). The vendored default is
    # worker_gpu_assignments=[1]*N (their multi-GPU rig put the router on cuda:0 and
    # workers on cuda:1+), which on one GPU raises "CUDA error: invalid device
    # ordinal". Pin every worker + the router to cuda:0. The Qwen3-0.6B router (×N
    # spawned copies) is tiny (~1.2GB each) — fits L40S 46GB even at 8 workers.
    infra = RouterInfrastructure(
        task=args.task, model_name="Qwen/Qwen3-0.6B", llm_names=llm_names,
        # 8192 (not the upstream 4096): reasoning workers (qwen3-32b-reasoning,
        # deepseek-r1) can burn the whole 4096 budget on the reasoningContent block
        # alone (stopReason=max_tokens) → empty answer → fugu 15× retry hang. Double
        # the budget so the answer text survives. _extract_text also falls back to
        # reasoning text as a belt-and-suspenders anti-hang.
        log_dir=str(log_dir), seed=42, temperature=0.1, max_tokens=8192, max_turns=5,
        servers={n: None for n in llm_names}, ports={n: None for n in llm_names},
        num_workers=args.num_workers, debug=args.debug,
        worker_gpu_assignments=[0] * args.num_workers,
        test_ratio=0.2, valid_ratio=0.2, configure_splits=True,
        trinity=True,
    )
    trainer = CMATrainingLoop(
        infrastructure=infra, num_iters=iters,
        test_interval=0 if smoke else 5,   # no validation run during smoke
        num_repeats=num_repeats, sigma0=0.03, seed=42, num_tests=300,
        test_size=300, servers={n: None for n in llm_names},
        opt_layer_indices=[26], popsize_override=popsize,
        diversity_bonus_weight=0.15, cost_bonus_weight=args.cost_bonus_weight,
        turn_bonus_weight=0.1, role_bonus_weight=0.0,
        use_structured_router=False, closed_model_config=None,
        agent_configs=cfgs, use_consultant=False, use_verifier=True,
        trinity=True, last_token_predict=False,
    )
    # Budget-constrained Pareto reward (set as attrs — vendored __init__ doesn't
    # accept these). reward = raw_passrate − λ·max(0, episode_cost − budget).
    # budget=0 or λ=0 → reward == raw pass-rate (upstream behavior preserved).
    trainer.cost_budget_usd = args.cost_budget_usd
    trainer.cost_lambda = args.cost_lambda
    if args.cost_budget_usd > 0 and args.cost_lambda > 0:
        _log(f"COST-AWARE reward: budget=${args.cost_budget_usd:.4f}/episode "
             f"lambda={args.cost_lambda} (real measured tokens × verified prices)")

    # Per-iteration callback: checkpoint + rollouts + telemetry to S3 (incl iter 0),
    # cost-cap enforcement, and (smoke) Phase-0.5 gates.
    def on_iter(it: int, iter_stats: dict) -> None:
        # Spend + throttle accumulate inside the spawned Pool workers (separate
        # interpreters), NOT this process. Aggregate across per-PID sink files
        # written under CAR_TRINITY_TELEMETRY_DIR (cost_bedrock/_Telemetry flush
        # after every call) so the cost cap actually sees real spend. (lessons #20)
        tele = bedrock_clients.aggregate_throttle()
        spend = cost_bedrock.aggregate_spend()
        iter_stats = dict(iter_stats or {})
        iter_stats["throttle"] = tele
        iter_stats["spend_usd"] = spend
        s3_put_json(iter_stats, args.s3_uri, f"iter_stats/iter_{it}.json")
        s3_sync(log_dir, args.s3_uri)   # checkpoints + rollouts, every iter incl 0
        # iter-0 durability HARD gate: prove the S3 sync path actually works before
        # committing the multi-day run. cost-aware-routing lost ~$200 to a spot
        # reclaim between iter 0 and the first persisted checkpoint; the lesson is
        # "confirm the iter-0 object EXISTS in S3, don't assume the sync worked."
        if it == 0 and args.s3_uri:
            es_key = args.s3_uri.rstrip("/") + "/es_log.json"
            try:
                subprocess.run(["aws", "s3", "ls", es_key], check=True,
                               capture_output=True, text=True, timeout=60)
                _log(f"iter-0 durability OK: {es_key} present in S3")
            except Exception:
                raise SystemExit(
                    f"FATAL: iter-0 checkpoint NOT in S3 ({es_key}) — the durability "
                    f"path is broken; refusing to run a multi-day job that can't "
                    f"survive a reclaim. Fix S3 sync before relaunching.")
        _log(f"iter {it}: spend=${spend:.2f} throttle={tele}")
        if spend >= args.cost_cap_usd:
            raise SystemExit(f"COST CAP ${args.cost_cap_usd} reached at iter {it} "
                             f"(spend ${spend:.2f}) — halting per pre-registered cap.")
        if tele["dropped_rate"] > 0.02:
            _log(f"WARNING dropped-episode rate {tele['dropped_rate']:.3%} > 2% — "
                 f"rate-limited, not compute-bound. Raise concurrency or add a region.")

    # Phase-0.5 smoke gate: wrap the callback to enforce the spec's non-degeneracy
    # checks (worker + role diversity, throttle survival, entropy) and halt early
    # if the training path is unhealthy BEFORE spending the full budget.
    def on_iter_gated(it: int, iter_stats: dict) -> None:
        on_iter(it, iter_stats)
        if smoke:
            ok, reasons = check_phase05_gates(iter_stats)
            if not ok and it >= 2:   # give 0,1,2 to warm up, then assert
                # Write the gate report BEFORE raising — the SystemExit skips the
                # final report-write block below, so persist it here for durability.
                agg = _collect_smoke_stats(log_dir)
                s3_put_json({"passed": False, "failures": reasons,
                             "halted_at_iter": it, "stats": agg},
                            args.s3_uri, "phase05_gate_report.json")
                raise SystemExit(f"PHASE 0.5 GATE FAIL at iter {it}: {reasons}")

    rc = trainer.train(iteration_callback=on_iter_gated)

    # Final exfiltration (artifact durability before any teardown).
    s3_sync(log_dir, args.s3_uri)
    s3_put_json(cost_bedrock.aggregate_cost_summary(), args.s3_uri, "cost_summary_final.json")

    if smoke:
        stats = _collect_smoke_stats(log_dir)
        ok, failures = check_phase05_gates(stats)
        s3_put_json({"passed": ok, "failures": failures, "stats": stats},
                    args.s3_uri, "phase05_gate_report.json")
        if not ok:
            _log("PHASE 0.5 GATES FAILED — DO NOT launch the full run:")
            for f in failures:
                _log(f"  ✗ {f}")
            return 3
        _log("PHASE 0.5 GATES PASSED — full run is cleared.")
    return rc or 0


def _train_with_polling(trainer, log_dir: Path, on_iter, *, smoke: bool) -> int:
    """Fallback driver: run trainer.train() and sync es_log after, emitting a
    per-iter sync by tailing es_log.json. Used only if the trainer lacks a
    callback hook. Best-effort — the per-iter S3 sync still happens via a thread."""
    import threading

    stop = threading.Event()

    def poller():
        seen = -1
        while not stop.wait(20):
            esl = log_dir / "es_log.json"
            if not esl.exists():
                continue
            try:
                entries = json.loads(esl.read_text())
            except Exception:  # noqa: BLE001
                continue
            it = len(entries)
            if it != seen:
                seen = it
                try:
                    on_iter(it, entries[-1] if entries else {})
                except SystemExit:
                    stop.set()
                    raise

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    try:
        rc = trainer.train()
    finally:
        stop.set()
    return rc or 0


def _collect_smoke_stats(log_dir: Path) -> dict:
    """Aggregate worker/role/throttle stats from the run's es_log for the gates."""
    esl = log_dir / "es_log.json"
    if not esl.exists():
        return {}
    entries = json.loads(esl.read_text())
    agent_distribution: dict = {}
    role_usage = {"solver": 0, "thinker": 0, "verifier": 0}
    verifier_early_halts = 0
    throttle = {"dropped": 0, "dropped_rate": 0.0}
    for e in entries:
        for name, c in (e.get("agent_distribution") or {}).items():
            agent_distribution[name] = agent_distribution.get(name, 0) + (c or 0)
        for r, c in (e.get("role_usage") or {}).items():
            if r in role_usage:
                role_usage[r] += c or 0
        verifier_early_halts += e.get("verifier_early_halts", 0) or 0
        t = e.get("throttle") or {}
        throttle["dropped"] += t.get("dropped", 0)
        throttle["dropped_rate"] = max(throttle["dropped_rate"], t.get("dropped_rate", 0.0))
    return {
        "agent_distribution": agent_distribution,
        "role_usage": role_usage,
        "verifier_early_halts": verifier_early_halts,
        "throttle": throttle,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Trinity agent-runtime entry point")
    p.add_argument("--phase", choices=["eval", "smoke", "train"], required=True)
    p.add_argument("--vendor-root", default="vendor/trinity-upstream")
    p.add_argument("--s3-uri", default=os.environ.get("TRINITY_S3_URI", ""))
    p.add_argument("--log-dir", default=None)
    p.add_argument("--model-file", default=None)
    p.add_argument("--task", default="livecodebench",
                   help="livecodebench (focused) or mix_m_m_r_l (paper's bundled mix)")
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=16,
                   help="local episode-eval parallelism (Bedrock latency is the bottleneck)")
    p.add_argument("--test-size", type=int, default=175)
    p.add_argument("--cost-cap-usd", type=float, default=5000.0)
    p.add_argument("--cost-bonus-weight", type=float, default=0.0,
                   help="0.0 reproduces upstream; sweep >0 for the cost-aware extension (OQ3)")
    p.add_argument("--cost-budget-usd", type=float, default=0.0,
                   help="per-episode $ budget for the budget-constrained Pareto reward; "
                        "0 disables cost shaping (raw pass-rate). Sweep with --cost-lambda.")
    p.add_argument("--cost-lambda", type=float, default=0.0,
                   help="penalty weight on per-episode cost overage above --cost-budget-usd. "
                        "Sweep this to trace the cost/accuracy Pareto frontier.")
    p.add_argument("--routing-policy", default=os.environ.get("CAR_TRINITY_ROUTING_POLICY", "learned"),
                   help="learned (Trinity head) | static:<ord> (best-static+scaffold) | "
                        "random (random+scaffold). Scaffold-matched baselines that hold the "
                        "3-role multi-turn loop fixed and vary ONLY routing — de-confounds the "
                        "routing claim from the verifier-scaffold effect.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    # Set BEFORE any install so it propagates to spawned workers (sitecustomize).
    os.environ["CAR_TRINITY_ROUTING_POLICY"] = args.routing_policy

    _log(f"phase={args.phase} vendor={args.vendor_root} s3={args.s3_uri or '(none)'} "
         f"routing={args.routing_policy}")
    if args.phase == "eval":
        return run_eval(args)
    return run_training(args, smoke=(args.phase == "smoke"))


if __name__ == "__main__":
    raise SystemExit(main())
