"""Spawn-safe Bedrock adaptation installer.

fugu's JobManager uses `multiprocessing.set_start_method("spawn")` + a Pool whose
workers re-import `fugu.utils` FRESH — so a monkeypatch applied only in the main
process (run_trinity_agent.install_bedrock_adaptation) never reaches the spawned
workers. Those workers then call the original OpenAI/Together/Gemini clients,
which fail (CLOSE-WAIT) and spin in the backoff loop → the run stalls.

Python runs `sitecustomize` automatically at interpreter startup for EVERY
process (including spawned multiprocessing workers) as long as this directory is
on PYTHONPATH. So we install the Bedrock Converse dispatch + pricing here, in
every worker, idempotently. This touches no vendored code (job_manager.py /
es.py stay as-is, per the spec's "fixed files" contract).

Activated by exporting PYTHONPATH=<scripts-dir>:$PYTHONPATH and
CAR_TRINITY_VENDOR_ROOT=<vendor-root> before launching.
"""
import os
import sys


def _install_bedrock_once() -> None:
    if os.environ.get("CAR_TRINITY_BEDROCK_PATCH") != "1":
        return
    if getattr(sys.modules.get(__name__), "_car_installed", False):
        return
    vendor_root = os.environ.get("CAR_TRINITY_VENDOR_ROOT")
    if vendor_root and vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    try:
        import fugu.utils   # noqa: F401  — ensure target module objects exist
        import fugu.cost    # noqa: F401
        import bedrock_clients
        import cost_bedrock
        import routing_policy
        bedrock_clients.install()
        cost_bedrock.install()
        routing_policy.install()   # baseline routing override (no-op when policy=learned)
        sys.modules[__name__]._car_installed = True
    except Exception as e:  # never let a worker die on import; surface in stderr
        sys.stderr.write(f"[sitecustomize] Bedrock patch install failed: {e}\n")


_install_bedrock_once()
