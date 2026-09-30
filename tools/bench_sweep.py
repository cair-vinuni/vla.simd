#!/usr/bin/env python3
"""Per-device benchmark sweep behind the reports in docs/benchmark/.

Run from the vla.simd checkout, after building it, with an interpreter that has
NumPy and huggingface_hub:

    python tools/bench_sweep.py --models-dir ~/models/bench --out results --threads 2,4,6,8,12

For every model and precision: a thread sweep, a coordinate search over the
runtime settings that leave results unchanged (a candidate is kept only when it
is at least 2% faster twice, the second time against a fresh baseline), a
greedy INT8 layer-group search, then a final measurement of the winner.
Every run is a fresh `vla_simd.policy_server --bench` process; each result is
appended to <out>/runs.jsonl as it lands, so an interrupted sweep keeps its data.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
MODELS = {
    "act": {"args": ["--model", "act"], "file": "act-so101-multi-task.gguf", "int8": 63},
    "impact": {"args": ["--model", "impact"], "file": "impact-so101-multi-task.gguf", "int8": 63,
               "int8_file": "impact-int8-so101-multi-task.gguf"},
    "smolvla": {"args": ["--model", "smolvla"], "file": "smolvla-so101-multi-task.gguf", "int8": 63},
    "octo": {"args": ["--model", "octo", "--cams", "front,wrist"],
             "file": "octo-small-so101-multi-task.gguf", "int8": 63},
    "turbovla": {"args": ["--model", "turbovla", "--task", TASK], "file": "turbovla-libero-f32.gguf",
                 "int8": None},
    "diffusion": {"args": ["--model", "diffusion"], "file": "diffusion-so101-tape.gguf", "int8": 3},
}
CONV = ("act", "impact", "diffusion", "octo")
VIEWS = ("act", "impact", "smolvla")


def candidates(model, backend, threads, precision, hybrid):
    """Runtime settings worth trying for this model on this backend. Each entry
    is merged into the current best configuration; none changes the results."""
    c = []
    if not hybrid and backend != "apple":
        c.append({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    if model in CONV:
        c += [{"TCPU_CONV_BUDGET": str(b)} for b in (32768, 131072, 524288)]
        c += [{"TCPU_CONV_MINP": str(p)} for p in (8, 32)]
    if model in VIEWS and threads >= 4:
        c.append({"TCPU_VIEW_THREADS": str(threads // 2)})
    if model == "smolvla":
        c += [{"TCPU_EXPERT_THREADS": str(t)} for t in (4, 8) if t < threads]
    if model == "octo":
        c += [{"TCPU_HEAD_THREADS": str(t)} for t in (0, 2, 4, 8) if t <= threads]
    c += [{"TCPU_OMP_MIN": str(v)} for v in (2048, 32768)]
    if backend == "neon":
        c += [{"TCPU_GEMM_SCHED": "static"}, {"TCPU_ATTN_QB": "8"}, {"TCPU_ATTN_QB": "32"}]
    if backend == "apple":
        c += [{"TCPU_ACCEL": "0"}, {"TCPU_ATTN_DYNAMIC": "0"}, {"TCPU_ATTN_DYNAMIC": "16"}]
    if precision == "int8":
        c += [{"TCPU_I8_MR": str(m)} for m in (2, 4, 6)]
        c += [{"TCPU_I8_MBLOCK": str(b)} for b in (0, 64, 256)]
    # drop settings equal to a backend default
    defaults = {"TCPU_CONV_BUDGET": "32768" if backend == "neon" else "131072",
                "TCPU_CONV_MINP": "16", "TCPU_OMP_MIN": "8192",
                "TCPU_I8_MR": "5" if backend in ("x86-avx2", "amd-zen") else "4",
                "TCPU_I8_MBLOCK": "128", "TCPU_HEAD_THREADS": "4" if backend == "apple" else "0"}
    return [e for e in c if not all(defaults.get(k) == v for k, v in e.items())]


class Runner:
    def __init__(self, a):
        self.a = a
        self.log = open(os.path.join(a.out, "runs.jsonl"), "a", encoding="utf-8")
        self.base_env = {k: v for k, v in os.environ.items()
                         if not k.startswith(("TCPU_", "OMP_", "DP_", "KMP_", "GOMP_"))
                         and not k.endswith("_INT8") and k != "SMOLVLA_NUM_STEPS"}
        self.base_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    def run(self, phase, model, precision, threads, env, mask, warmup, queries, affinity=None):
        spec = MODELS[model]
        f = spec.get("int8_file", spec["file"]) if precision == "int8" else spec["file"]
        cmd = [sys.executable, "-m", "vla_simd.policy_server", *spec["args"],
               "--model-dir", os.path.join(self.a.models_dir, f),
               "--warmup", str(warmup), "--bench", str(queries), "--json"]
        if precision == "int8":
            cmd += ["--int8", str(mask)]
        if affinity:
            cmd = ["taskset", "-c", affinity] + cmd
        e = dict(self.base_env, OMP_NUM_THREADS=str(threads), **env)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=self.a.timeout)
            out = [l for l in p.stdout.splitlines() if l.startswith("{")]
            res = json.loads(out[-1]) if out else None
            err = None if res else (p.stderr.strip().splitlines() or ["no output"])[-3:]
        except subprocess.TimeoutExpired:
            res, err = None, ["timeout"]
        rec = {"phase": phase, "model": model, "precision": precision, "threads": threads,
               "env": env, "int8_mask": mask if precision == "int8" else None,
               "affinity": affinity, "warmup": warmup, "queries": queries,
               "wall_s": round(time.time() - t0, 1), "result": res, "error": err,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        med = res["median_ms"] if res else float("inf")
        print(f"[{rec['time']}] {phase:7s} {model:9s} {precision:4s} t={threads:<2d} "
              f"{env or ''} {('mask=%s' % mask) if precision == 'int8' else ''} "
              f"{('cpus=' + affinity) if affinity else ''} -> "
              f"{med if res else err}", flush=True)
        return med, res


def build_info():
    """Commit and compiler of the build the server loads (build/)."""
    info = {}
    try:
        info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()
        info["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                            capture_output=True, text=True).stdout.strip())
        cache = dict(l.split("=", 1) for l in open(os.path.join("build", "CMakeCache.txt"))
                     if "=" in l and not l.startswith(("#", "//")))
        cxx = cache.get("CMAKE_CXX_COMPILER:FILEPATH", "").strip()
        info["cxx"] = subprocess.run([cxx, "--version"], capture_output=True,
                                     text=True).stdout.splitlines()[0]
        info["build_type"] = cache.get("CMAKE_BUILD_TYPE:STRING", "").strip()
        info["neon_mr"] = cache.get("VLA_NEON_MR:STRING", "").strip() or None
    except (OSError, IndexError, ValueError) as e:
        info["error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", required=True, help="comma list for the thread sweep")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--pcores", default="", help="Linux CPU list of one thread per P-core (hybrid x86)")
    ap.add_argument("--sweep-queries", type=int, default=20)
    ap.add_argument("--final-queries", type=int, default=50)
    ap.add_argument("--slow-ms", type=float, default=3000.0,
                    help="above this median the final run uses 20 queries")
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()
    a.models_dir = os.path.expanduser(a.models_dir)
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.out, exist_ok=True)
    r = Runner(a)
    threads = [int(t) for t in a.threads.split(",")]
    hybrid = bool(a.pcores)
    n_pcores = len(a.pcores.split(",")) if hybrid else 0
    summary = []
    info = dict(build_info(), started=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    for model in a.models.split(","):
        spec = MODELS[model]
        precisions = ["fp32"] + (["int8"] if spec["int8"] else [])
        for precision in precisions:
            mask = spec["int8"]
            # 1. threads
            sweep = {}
            backend, int8_ok = None, True
            for t in threads:
                med, res = r.run("threads", model, precision, t, {}, mask, 5, a.sweep_queries)
                sweep[t] = med
                if res:
                    backend = res["backend"]
                elif precision == "int8" and t == threads[0]:
                    int8_ok = False
                    break
            if not int8_ok or backend is None:
                summary.append({"model": model, "precision": precision, "status": "failed"})
                continue
            best_t = min(sweep, key=sweep.get)
            best = {"threads": best_t, "env": {}, "mask": mask, "affinity": None}
            best_med = sweep[best_t]
            # 2. P-core pinning on hybrid x86 (Linux)
            if hybrid and best_t >= n_pcores:
                med, _ = r.run("affinity", model, precision, n_pcores, {}, mask, 5, a.sweep_queries,
                               affinity=a.pcores)
                if med < 0.98 * best_med:
                    confirm, _ = r.run("confirm", model, precision, n_pcores, {}, mask, 5,
                                       a.sweep_queries, affinity=a.pcores)
                    base, _ = r.run("confirm", model, precision, best["threads"], {}, mask, 5,
                                    a.sweep_queries)
                    if confirm < 0.98 * base:
                        best.update(threads=n_pcores, affinity=a.pcores)
                        best_med = confirm
            # 3. runtime settings, coordinate search
            for cand in candidates(model, backend, best["threads"], precision, hybrid):
                env = dict(best["env"], **cand)
                med, _ = r.run("knob", model, precision, best["threads"], env, best["mask"], 5,
                               a.sweep_queries, best["affinity"])
                if med < 0.98 * best_med:
                    confirm, _ = r.run("confirm", model, precision, best["threads"], env, best["mask"],
                                       5, a.sweep_queries, best["affinity"])
                    base, _ = r.run("confirm", model, precision, best["threads"], best["env"],
                                    best["mask"], 5, a.sweep_queries, best["affinity"])
                    if confirm < 0.98 * base:
                        best["env"] = env
                        best_med = confirm
            # 4. INT8 layer groups: drop one group at a time while that is faster
            if precision == "int8":
                for bit in (1, 2, 4, 8, 16, 32):
                    if not best["mask"] & bit or best["mask"] == bit:
                        continue
                    m = best["mask"] & ~bit
                    med, _ = r.run("mask", model, precision, best["threads"], best["env"], m, 5,
                                   a.sweep_queries, best["affinity"])
                    if med < 0.98 * best_med:
                        confirm, _ = r.run("confirm", model, precision, best["threads"], best["env"], m,
                                           5, a.sweep_queries, best["affinity"])
                        base, _ = r.run("confirm", model, precision, best["threads"], best["env"],
                                        best["mask"], 5, a.sweep_queries, best["affinity"])
                        if confirm < 0.98 * base:
                            best["mask"] = m
                            best_med = confirm
            # 5. final measurement
            q = a.final_queries if best_med < a.slow_ms else 20
            med, res = r.run("final", model, precision, best["threads"], best["env"], best["mask"], 5, q,
                             best["affinity"])
            summary.append({"model": model, "precision": precision, "best": best, "sweep": sweep,
                            "final": res})
            with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"host_platform": platform.platform(), "python": sys.version.split()[0],
                           "build": info, "summary": summary}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
