"""Create or update the Hugging Face Space for the demo (Docker SDK, cpu-basic, public) and upload exactly what the service
needs: app.py, the uplift package, the three served bundles, the dev serving files, the test-split plots and tables the UI
reads, and space/README.md as the Space README. Then wait for RUNNING, show the build log, and smoke-test the live app.
Never prints a token: it relies on the logged-in session (`hf auth login` or HF_TOKEN). Space variables APP_VERSION and BUDGET
are set from the git short hash and 0.10.

Usage: python scripts/deploy_space.py [--repo DavutKursun/uplift-targeting] [--wait-minutes 25] [--no-smoke] [--calls 30]
Writes eval/results/latency_space.json from the live /score calls.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uplift.config import FEATURES

FILES = [
    "app.py", "requirements-space.txt", "Dockerfile", ".dockerignore",
    "uplift/__init__.py", "uplift/config.py", "uplift/schema.py", "uplift/serving.py", "uplift/ui.py", "uplift/learners.py", "uplift/policy.py",
    "uplift/metrics.py", "uplift/data.py", "uplift/synthetic.py",
    "models/s_learner-visit.joblib", "models/r_learner-visit.joblib", "models/response-visit.joblib",
    "models/serving/policy_grid_dev_visit.json", "models/serving/customers_dev_sample.csv", "models/serving/version.json",
    "eval/results/qini_curves_test.png", "eval/results/deciles_test_s_learner.png", "eval/results/budget_curve_test.png", "eval/results/profit_curve_test.png",
    "eval/results/paired_test_visit.json", "eval/results/placebo_null_dev_visit_s_learner.json", "eval/results/placebo_dev_visit_s_learner.json",
    "eval/results/placebo_test_visit_s_learner.json", "eval/results/dev_vs_test.md",
]
SECRET = re.compile(r"(hf_[A-Za-z0-9]{10,}|gh[po]_[A-Za-z0-9]{10,})")


def redact(s: str) -> str:
    return SECRET.sub("[redacted]", s)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=3, check=False).stdout.strip() or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"


def stage(tmp: Path) -> None:
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(rel)
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(ROOT / "space" / "README.md", tmp / "README.md")


def build_log(api, repo: str, lines: int = 80) -> list[str]:
    from huggingface_hub.utils import build_hf_headers

    out: list[str] = []
    try:
        with httpx.stream("GET", f"https://huggingface.co/api/spaces/{repo}/logs/build", headers=build_hf_headers(), timeout=20.0) as r:
            for raw in r.iter_lines():
                if raw.startswith("data:"):
                    try:
                        d = json.loads(raw[5:].strip())
                        out.append(redact(str(d.get("data", d))))
                    except json.JSONDecodeError:
                        out.append(redact(raw[5:].strip()))
                if len(out) > 2000:
                    break
    except (httpx.HTTPError, httpx.StreamError) as e:
        out.append(f"(build log unavailable: {type(e).__name__})")
    return out[-lines:]


def smoke(base: str, calls: int, out: Path, version: str) -> dict:
    sample = pd.read_csv(ROOT / "models" / "serving" / "customers_dev_sample.csv")
    body = {f: float(sample.iloc[0][f]) for f in FEATURES}
    with httpx.Client(base_url=base, timeout=60.0) as http:
        health = None
        for _ in range(60):
            try:
                r = http.get("/health")
                if r.status_code == 200:
                    health = r.json()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        if health is None:
            raise RuntimeError("the live /health never answered 200")
        print("health:", json.dumps({k: health[k] for k in ("status", "models", "budget", "version")}))
        scores = {}
        for model in ("s_learner", "r_learner"):
            r = http.post(f"/score?model={model}", json=body)
            assert r.status_code == 200, r.text
            scores[model] = r.json()
            print(f"score {model}: uplift {scores[model]['uplift']:+.4f} segment {scores[model]['segment']} decision {scores[model]['decision']} latency_ms {scores[model]['latency_ms']}")
        pol = http.get("/policy?budget=0.10").json()
        print(f"policy 10%: threshold {pol['threshold']:.5f}, {pol['incremental_per_1000']:.1f} raw / {pol['ipw_per_1000']:.1f} IPW per 1,000, break-even {pol['breakeven_budget']}")
        lat = {}
        for model in ("s_learner", "r_learner"):
            ms = []
            for _ in range(calls):
                t0 = time.perf_counter()
                r = http.post(f"/score?model={model}", json=body)
                ms.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200
            a = np.asarray(ms)
            lat[model] = {"n": calls, "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
                          "server_latency_ms_last": r.json()["latency_ms"]}
            print(f"live {model}: round-trip p50 {lat[model]['p50_ms']} ms, p95 {lat[model]['p95_ms']} ms over {calls} calls (server-side {lat[model]['server_latency_ms_last']} ms)")
    result = {"space": base, "version_deployed": version, "health": health, "round_trip": lat, "note": "round-trip from a laptop to the Space (network included); server-side latency_ms is in each response",
              "policy_10pct": {k: pol[k] for k in ("threshold", "incremental_per_1000", "ipw_per_1000", "profit_per_1000_contacts", "breakeven_budget")}}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="DavutKursun/uplift-targeting")
    ap.add_argument("--wait-minutes", type=float, default=25)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "latency_space.json")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi

    api = HfApi()
    print("logged in as", api.whoami()["name"])
    sha = git_sha()
    url = api.create_repo(repo_id=a.repo, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", private=False, exist_ok=True)
    print("space:", redact(str(url)))
    api.add_space_variable(a.repo, "APP_VERSION", sha)
    api.add_space_variable(a.repo, "BUDGET", "0.10")
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        info = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="space", commit_message=f"deploy {sha}")
        print("uploaded:", redact(str(getattr(info, "commit_url", info))))
    owner, name = a.repo.split("/")
    base = f"https://{owner.lower()}-{name.lower()}.hf.space"
    deadline = time.time() + a.wait_minutes * 60
    last = None
    while time.time() < deadline:
        stage_ = api.get_space_runtime(a.repo).stage
        if stage_ != last:
            print(f"{time.strftime('%H:%M:%S')} stage {stage_}", flush=True)
            last = stage_
        if stage_ == "RUNNING":
            break
        if stage_ in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "DELETING"):
            print("\n".join(build_log(api, a.repo)))
            return 1
        time.sleep(15)
    else:
        print("timed out waiting for RUNNING")
        return 1
    print("--- build log (tail) ---")
    print("\n".join(build_log(api, a.repo, 40)))
    if not a.no_smoke:
        smoke(base, a.calls, a.out, sha)
        print(f"wrote {a.out}")
    print("live:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
