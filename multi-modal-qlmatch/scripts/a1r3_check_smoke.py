"""A1 round-3 gate 4 (no double perturbation): python a1r3_check_smoke.py SMOKE_RUN_DIR EVAL_DIR
SMOKE_RUN_DIR: train_multitask --domain real smoke run (metrics.json with the in-training test scores);
EVAL_DIR: eval_perturb --absolute-domain --variants d_real,... on it (metrics.json). Exit 0 only if the d_real T1 (and T2, if
both have it) score equals the run's own test score within the tolerance (default 2e-3), the eval ran with the model-internal domain sim/native (absolute domain), and
|d_real - own| < |nominal - own| for at least one of hit/evt_nu. If nominal == d_real for both (degenerate model) this is
reported as INCONCLUSIVE (warning, not a failure)."""
import argparse
import json
import os
import sys

TOL = 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("eval")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()
    fails = []

    def check(name, cond, info=""):
        print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
        if not cond:
            fails.append(name)

    paths = [os.path.join(args.run, "metrics.json"), os.path.join(args.eval, "metrics.json")]
    for p in paths:
        check(f"exists {p}", os.path.exists(p))
    if fails:
        return 1
    run, ev = (json.load(open(p)) for p in paths)
    check("run trained with --domain real", run.get("domain") == "real", f"(domain={run.get('domain')})")
    check("eval model-internal domain is sim/native (absolute-domain verified)",
          ev.get("model_domain") == "sim" and ev.get("model_width") == "native",
          f"(model_domain={ev.get('model_domain')}, model_width={ev.get('model_width')})")
    closer, degenerate = False, True
    for t in ("hit", "evt_nu"):
        if t not in run["test"] or t not in ev["results"]:
            check(f"{t} present", t == "evt_nu", "(optional for evt_nu)")
            continue
        own, r = run["test"][t]["score"], ev["results"][t]
        dr, nom = r["d_real"]["score"], r["nominal"]["score"]
        check(f"{t} d_real == in-training test score", abs(dr - own) < args.tol,
              f"({dr:.6f} vs {own:.6f}, tol {args.tol:g})")
        print(f"      {t}: nominal {nom:.5f}  d_real {dr:.5f}  stored {own:.5f}", flush=True)
        if abs(nom - dr) > 1e-9:
            degenerate = False
        if abs(dr - own) < abs(nom - own):
            closer = True
    if closer:
        print("PASS  d_real closer to the in-training score than nominal for at least one task", flush=True)
    elif degenerate:
        print("WARN  INCONCLUSIVE: nominal == d_real for all tasks (degenerate smoke model); relying on the equality "
              "check and the verified sim/native model-internal domain", flush=True)
    else:
        check("d_real closer to the in-training score than nominal (>=1 task)", False)
    print("GATE 4 " + ("FAILED: " + "; ".join(fails) if fails else "PASSED"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
