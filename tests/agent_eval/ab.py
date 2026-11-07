"""Repeated A/B runs of the eval suite: the old harness (control, DSH_HARNESS_BOOST=0) against the new one (boost).

Runs arms one after another, alternating, so slow drift (server state, temperature) hits both. Before each arm, dsh
is restarted with DSH_HARNESS_BOOST set for that arm, and the arm gets freshly generated task folders. driver.js must
be running in the dsh page (it survives restarts and waits between runs). At the end dsh is restarted with the boost on.

    python tests/agent_eval/ab.py --families mini_interp,semver_range,planted_fail,long_spec --seeds 1,2 \\
        --arms control-2,boost-2,control-3,boost-3,boost-4
    python tests/agent_eval/run_eval.py report baseline control-2 control-3 boost-2 boost-3 boost-4
    python tests/agent_eval/ab.py --compare baseline,control-2,control-3 --vs boost-2,boost-3,boost-4

An arm named control-* runs with the boost off, any other name with it on.
"""
import argparse, json, os, subprocess, sys, time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402

START_DSH = r"C:\src\shoehorn\scripts\windows-harness\start-dsh.ps1"


def pwsh(script):
    return subprocess.run(["pwsh", "-NoProfile", "-Command", script], capture_output=True, text=True)


def restart_dsh(boost_on):
    """Stop the dsh listening on 127.0.0.1:3080 (that process only) and start it again with the arm's setting."""
    value = "1" if boost_on else "0"
    pwsh("$c = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue;"
         " if ($c) { Stop-Process -Id $c.OwningProcess -Force }")
    time.sleep(3)
    pwsh(f"$env:DSH_HARNESS_BOOST = '{value}'; Start-Process pwsh -ArgumentList '-NoProfile','-File','{START_DSH}' -WindowStyle Hidden")
    deadline = time.time() + 180
    while time.time() < deadline:
        r = pwsh("[bool](Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue)")
        if r.stdout.strip() == "True":
            time.sleep(15)  # loader settles; the page reconnects
            return
        time.sleep(3)
    raise RuntimeError("dsh did not come back on 127.0.0.1:3080")


def run_arm(name, families, seeds, timeout_min):
    boost_on = not name.startswith("control")
    root = rf"C:\tmp\agent-eval\ab\{name}"
    key = rf"C:\tmp\agent-eval\ab-key\{name}"
    print(f"=== {name} (boost {'on' if boost_on else 'off'})", flush=True)
    restart_dsh(boost_on)
    subprocess.run([sys.executable, os.path.join(HERE, "gen_tasks.py"), "--tier", "all", "--seeds", seeds,
                    "--only", families, "--root", root, "--key", key], check=True)
    args = SimpleNamespace(name=name, key=key, preset="minimal-compact", only="", timeout_min=timeout_min,
                           note=f"A/B arm {name}: DSH_HARNESS_BOOST={'1' if boost_on else '0'}")
    run_eval.Runner(args).serve()


def compare(control, boost):
    """Per task: passes out of runs in each arm, and the totals; a task that passes less often with the boost is flagged."""
    a = [run_eval.load_run(n) for n in control]
    b = [run_eval.load_run(n) for n in boost]
    ids = sorted(set().union(*[set(r) for r in a + b]))
    print(f"{'task':18}{'control':>12}{'boost':>12}   steps c/b    wall s c/b")
    totals = [0, 0, 0, 0]
    for i in ids:
        ra = [r[i] for r in a if i in r]
        rb = [r[i] for r in b if i in r]
        pa, pb = sum(r.get("pass", False) for r in ra), sum(r.get("pass", False) for r in rb)
        totals[0] += pa
        totals[1] += len(ra)
        totals[2] += pb
        totals[3] += len(rb)
        mean = lambda rs, k: round(sum(r["metrics"][k] for r in rs if "metrics" in r) / max(1, len(rs)))
        flag = "  <-- worse" if len(ra) and len(rb) and pb / len(rb) < pa / len(ra) else ""
        print(f"{i:18}{pa:>8}/{len(ra):<3}{pb:>8}/{len(rb):<3}   {mean(ra, 'steps'):>4}/{mean(rb, 'steps'):<4}  "
              f"{mean(ra, 'wall_s'):>5}/{mean(rb, 'wall_s'):<5}{flag}")
    print(f"{'total':18}{totals[0]:>8}/{totals[1]:<3}{totals[2]:>8}/{totals[3]:<3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="mini_interp,semver_range,planted_fail,long_spec")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--arms", default="control-2,boost-2,control-3,boost-3,boost-4")
    ap.add_argument("--timeout-min", type=int, default=60)
    ap.add_argument("--compare", default="")
    ap.add_argument("--vs", default="")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare.split(","), a.vs.split(","))
        return
    try:
        for name in a.arms.split(","):
            run_arm(name, a.families, a.seeds, a.timeout_min)
    finally:
        restart_dsh(True)
        print("dsh restarted with the boost on", flush=True)


if __name__ == "__main__":
    main()
