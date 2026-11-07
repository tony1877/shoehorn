"""Run the synthetic eval suite through the real dsh web server and grade it.

The tasks run in ordinary dsh sessions, created by driver.js inside the dsh page (the page is already signed in), so
every run uses the live preset, persona, tools, hooks and model route. This script hands the driver one task at a
time over a local HTTP endpoint, watches the session log until the agent is idle, grades the workspace against the
hidden tests, and writes one result per task.

    python tests/agent_eval/gen_tasks.py --seeds 1,2
    python tests/agent_eval/run_eval.py run --name baseline [--preset minimal-compact] [--only ...] [--timeout-min 45]
        then paste tests/agent_eval/driver.js into the dsh page's devtools console (or let Claude inject it)
    python tests/agent_eval/run_eval.py report baseline [other-run]

Results: tests/agent_eval/results/<name>/<task>.json and summary.json (results/ is not committed).
"""
import argparse, http.server, json, os, re, shutil, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(HERE, "..", "..", "agent")
TSX = "file:///" + os.path.abspath(os.path.join(AGENT, "node_modules", "tsx", "dist", "esm", "index.mjs")).replace("\\", "/")
SESSIONS = os.path.join(os.environ.get("DSH_HOME") or os.path.join(os.path.expanduser("~"), ".dsh"), "sessions")
RESULTS = os.path.join(HERE, "results")
PORT = 3099
IDLE_SECONDS = 45          # no new events this long after a turn/end: the agent is done
CLAIM = re.compile(r"\b(all (\d+ )?tests? (now )?pass|tests? (now )?pass(es|ing)?|\d+ passed|\bOK\b.*tests?|"
                   r"suite passes|everything passes|fixed)\b", re.I)


# ---------------------------------------------------------------------------------------------------- session logs

def session_file(cwd, session_id):
    """The session log: sessions/<encoded cwd>/<id>/session.v4.jsonl.zstd (searched, the encoding is not ours)."""
    for d in os.listdir(SESSIONS):
        p = os.path.join(SESSIONS, d, session_id, "session.v4.jsonl.zstd")
        if os.path.exists(p):
            return p
    return None


def read_events(path):
    out = subprocess.run(["node", "--import", TSX, os.path.join(HERE, "read_session.mts"), path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def idle(events):
    """Whether the last turn has ended and no step is open."""
    last = None
    for e in events:
        if e.get("type") in ("turn/start", "turn/end"):
            last = e["type"]
    return last == "turn/end"


def metrics(events):
    m = {"turns": 0, "steps": 0, "tool_calls": 0, "tool_errors": 0, "edit_failures": 0, "shell_nonzero": 0,
         "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "compactions": 0, "harness_nudges": 0,
         "auto_continues": 0, "repeat_reminders": 0, "tools": {}, "wall_s": 0, "final_text": ""}
    names, first, last = {}, None, None
    for e in events:
        t, d = e.get("type"), e.get("data") or {}
        if "time" in e:
            first = first or e["time"]
            last = e["time"]
        if t == "turn/start":
            m["turns"] += 1
        elif t == "step/start":
            m["steps"] += 1
        elif t == "tool/call":
            m["tool_calls"] += 1
            names[d.get("callId")] = d.get("name")
            m["tools"][d.get("name")] = m["tools"].get(d.get("name"), 0) + 1
        elif t == "tool/result" and e.get("surfaceOp", "append") == "append":
            msg = d.get("message") or {}
            name = names.get(msg.get("toolCallId"))
            text = "".join(b.get("text", "") for b in msg.get("content", []) if isinstance(b, dict))
            if msg.get("isError"):
                m["tool_errors"] += 1
                if name in ("edit", "write", "str_replace_editor"):
                    m["edit_failures"] += 1
            if name in ("pwsh", "bash") and re.search(r"\[(exit code:|Command finished with exit code) [1-9]", text):
                m["shell_nonzero"] += 1
        elif t == "assistant/message":
            u = d.get("usage") or {}
            m["input_tokens"] += u.get("inputTokens", 0)
            m["output_tokens"] += u.get("outputTokens", 0)
            m["cache_read_tokens"] += u.get("cacheReadTokens", 0)
            texts = [b.get("text", "") for b in (d.get("message") or {}).get("content", []) if b.get("type") == "text"]
            if texts and "".join(texts).strip():
                m["final_text"] = "".join(texts).strip()
        elif t and t.startswith("compaction/") and t.endswith("summary"):
            m["compactions"] += 1
        elif t == "user/message":
            kind = (d.get("source") or {}).get("kind")
            if kind == "auto-continue":
                m["auto_continues"] += 1
            elif kind == "repeat-tool-reminder":
                m["repeat_reminders"] += 1
            text = "".join(b.get("text", "") for b in d.get("content", []) if isinstance(b, dict))
            if text.lstrip().startswith("[harness]"):
                m["harness_nudges"] += 1
    m["wall_s"] = round(((last or 0) - (first or 0)) / 1000)
    return m


# ---------------------------------------------------------------------------------------------------- grading

def run(cmd, cwd, timeout=300):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def unittest_counts(output):
    ran = re.search(r"Ran (\d+) tests?", output)
    fails = re.search(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?", output)
    n = int(ran.group(1)) if ran else 0
    bad = (int(fails.group(1) or 0) + int(fails.group(2) or 0)) if fails else (0 if ran else n)
    return n, bad


def grade(task):
    g = json.load(open(os.path.join(task["key"], "grade.json"), encoding="utf-8"))
    res = {"checks": {}}
    work = tempfile.mkdtemp(prefix="agent-eval-grade-")
    try:
        ws = os.path.join(work, "ws")
        shutil.copytree(task["cwd"], ws, ignore=shutil.ignore_patterns("__pycache__"))
        if g.get("hidden_tests"):
            os.makedirs(os.path.join(ws, "tests"), exist_ok=True)
            open(os.path.join(ws, "tests", "__init__.py"), "a").close()
            shutil.copy(os.path.join(task["key"], "hidden", "test_hidden.py"), os.path.join(ws, "tests", "test_zz_hidden.py"))
            code, out = run([sys.executable, "-m", "unittest", "tests.test_zz_hidden", "-v"], ws)
            n, bad = unittest_counts(out)
            res["checks"]["hidden"] = {"ran": n, "failed": bad, "ok": code == 0 and n > 0, "tail": out[-1500:]}
            os.remove(os.path.join(ws, "tests", "test_zz_hidden.py"))
            code, out = run([sys.executable, "-m", "unittest"], ws)
            n, bad = unittest_counts(out)
            res["checks"]["visible"] = {"ran": n, "failed": bad, "ok": code == 0 and n > 0, "tail": out[-800:]}
        if "exact_file" in g:
            ef = g["exact_file"]
            want = open(os.path.join(task["key"], ef["expected"]), encoding="utf-8").read()
            got = open(os.path.join(ws, ef["path"]), encoding="utf-8", newline="").read().replace("\r\n", "\n") \
                if os.path.exists(os.path.join(ws, ef["path"])) else None
            res["checks"]["exact_file"] = {"ok": got == want}
        if "pwsh_output" in g:
            po = g["pwsh_output"]
            out_file = os.path.join(ws, po["file"])
            if os.path.exists(out_file):
                os.remove(out_file)
            code, out = run(["pwsh", "-NoProfile", "-File", os.path.join(ws, po["script"])], ws)
            got = open(out_file, encoding="utf-8").read().strip() if os.path.exists(out_file) else None
            res["checks"]["pwsh_output"] = {"ok": got == po["expected"], "got": got, "expected": po["expected"]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res["pass"] = bool(res["checks"]) and all(c["ok"] for c in res["checks"].values())
    return res


# ---------------------------------------------------------------------------------------------------- run loop

class Runner:
    def __init__(self, a):
        suite = json.load(open(os.path.join(a.key, "suite.json"), encoding="utf-8"))
        only = [x for x in a.only.split(",") if x]
        self.tasks = [t for t in suite if not only or t["family"] in only or t["id"] in only]
        self.out = os.path.join(RESULTS, a.name)
        os.makedirs(self.out, exist_ok=True)
        self.done = {f[:-5] for f in os.listdir(self.out) if f.endswith(".json") and f != "summary.json"}
        self.preset, self.timeout = a.preset, a.timeout_min * 60
        self.note = a.note
        self.lock = threading.Lock()
        self.current = None      # {"task", "session", "started"}
        self.cancel = None

    def next_payload(self):
        with self.lock:
            if self.cancel:
                sid, self.cancel = self.cancel, None
                return {"cancel": sid}
            if self.current is not None:
                return {"wait": True}
            for t in self.tasks:
                if t["id"] not in self.done:
                    self.current = {"task": t, "session": None, "started": time.time()}
                    print(f"[{time.strftime('%H:%M:%S')}] -> {t['id']}", flush=True)
                    return {"task": {"id": t["id"], "cwd": t["cwd"], "prompt": t["prompt"], "preset": self.preset}}
            return {"finished": True}

    def started(self, body):
        with self.lock:
            if self.current and self.current["task"]["id"] == body.get("id"):
                if body.get("error"):
                    print(f"   driver error: {body['error']}", flush=True)
                    self.finish({"error": body["error"]}, [])
                else:
                    self.current["session"] = body["sessionId"]
                    self.current["started"] = time.time()

    def finish(self, extra, events):
        t = self.current["task"]
        res = {"id": t["id"], "family": t["family"], "tier": t.get("tier", "easy"), "seed": t["seed"], "session": self.current["session"],
               "preset": self.preset, **extra}
        if "error" not in extra:
            res.update(grade(t))
            res["metrics"] = metrics(events)
            final = res["metrics"].get("final_text", "")
            res["claims_success"] = bool(CLAIM.search(final))
            visible = res["checks"].get("visible")
            # false: the claim is wrong about what the agent could check itself (its visible tests, or the
            # task's own check); hidden_miss: the claim held for the visible tests but hidden edge cases fail
            res["false_success"] = res["claims_success"] and not res["pass"] and (visible is None or not visible["ok"])
            res["hidden_miss"] = res["claims_success"] and not res["pass"] and visible is not None and visible["ok"]
        json.dump(res, open(os.path.join(self.out, f"{t['id']}.json"), "w", encoding="utf-8"), indent=1)
        self.done.add(t["id"])
        print(f"   {t['id']}: {'PASS' if res.get('pass') else 'FAIL'}"
              f"{' (false success claim)' if res.get('false_success') else ''}"
              f"{' ' + extra['status'] if 'status' in extra else ''}", flush=True)
        self.current = None

    def watch(self):
        while True:
            time.sleep(10)
            with self.lock:
                cur = self.current
                if cur is None or cur["session"] is None:
                    continue
                path = session_file(cur["task"]["cwd"], cur["session"])
                if path is None:
                    continue
                age = time.time() - os.path.getmtime(path)
                if time.time() - cur["started"] > self.timeout:
                    self.cancel = cur["session"]
                    self.finish({"status": "timeout"}, read_events(path))
                    continue
                if age < IDLE_SECONDS:
                    continue
                events = read_events(path)
                if idle(events):
                    self.finish({"status": "completed"}, events)

    def serve(self):
        runner = self

        class H(http.server.BaseHTTPRequestHandler):
            def cors(self):
                self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Private-Network", "true")

            def reply(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self.cors()
                self.end_headers()

            def do_GET(self):
                self.reply(runner.next_payload() if self.path.startswith("/next") else {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path.startswith("/started"):
                    runner.started(body)
                self.reply({"ok": True})

            def log_message(self, *args):
                pass

        threading.Thread(target=self.watch, daemon=True).start()
        print(f"{len(self.tasks)} tasks, {len(self.done)} already done; waiting for driver.js on 127.0.0.1:{PORT}",
              flush=True)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        while True:
            time.sleep(5)
            with self.lock:
                if self.current is None and all(t["id"] in self.done for t in self.tasks):
                    break
        time.sleep(15)  # let the driver see "finished"
        srv.shutdown()
        srv.server_close()
        summarize(self.out, self.note)


# ---------------------------------------------------------------------------------------------------- reports

def load_run(name):
    d = os.path.join(RESULTS, name)
    return {r["id"]: r for r in (json.load(open(os.path.join(d, f), encoding="utf-8"))
                                 for f in sorted(os.listdir(d)) if f.endswith(".json") and f != "summary.json")}


def aggregate(rs):
    rs = [r for r in rs if "error" not in r]
    n = len(rs) or 1
    tot = lambda k: sum(r["metrics"][k] for r in rs)
    return {"tasks": len(rs), "pass": sum(r["pass"] for r in rs), "pass_rate": round(sum(r["pass"] for r in rs) / n, 3),
            "false_success": sum(r.get("false_success", False) for r in rs),
            "hidden_miss": sum(r.get("hidden_miss", False) for r in rs),
            "timeouts": sum(r.get("status") == "timeout" for r in rs),
            "mean_steps": round(tot("steps") / n, 1), "mean_wall_s": round(tot("wall_s") / n),
            "edit_failures": tot("edit_failures"), "shell_nonzero": tot("shell_nonzero"),
            "output_tokens": tot("output_tokens"), "compactions": tot("compactions"),
            "harness_nudges": tot("harness_nudges"), "auto_continues": tot("auto_continues"),
            "repeat_reminders": tot("repeat_reminders")}


def summarize(out, note=""):
    rs = list(load_run(os.path.basename(out)).values())
    s = {"note": note, "overall": aggregate(rs), "by_family": {}}
    for tier in sorted({r.get("tier", "easy") for r in rs}):
        s.setdefault("by_tier", {})[tier] = aggregate([r for r in rs if r.get("tier", "easy") == tier])
    for fam in sorted({r["family"] for r in rs}):
        s["by_family"][fam] = aggregate([r for r in rs if r["family"] == fam])
    json.dump(s, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(s["overall"], indent=1))


def report(names):
    runs = {n: load_run(n) for n in names}
    keys = ["pass_rate", "false_success", "hidden_miss", "timeouts", "mean_steps", "mean_wall_s", "edit_failures", "shell_nonzero",
            "output_tokens", "compactions", "harness_nudges"]
    print(f"{'':16}" + "".join(f"{n:>16}" for n in names))
    aggs = {n: aggregate(list(r.values())) for n, r in runs.items()}
    for k in keys:
        print(f"{k:16}" + "".join(f"{aggs[n][k]:>16}" for n in names))
    for tier in ("easy", "hard"):
        rates = [aggregate([r for r in runs[n].values() if r.get("tier", "easy") == tier]) for n in names]
        print(f"{'pass ' + tier:16}" + "".join(f"{a['pass']:>12}/{a['tasks']:<3}" for a in rates))
    print()
    ids = sorted(set().union(*[set(r) for r in runs.values()]))
    for i in ids:
        cells = []
        for n in names:
            r = runs[n].get(i)
            cells.append("-" if r is None else ("PASS" if r.get("pass") else "FAIL")
                         + ("!" if r and r.get("false_success") else "")
                         + (f" {r['metrics']['steps']}st" if r and "metrics" in r else ""))
        print(f"{i:16}" + "".join(f"{c:>16}" for c in cells))
    print("\n! = claimed success while failing")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--key", default=r"C:\tmp\agent-eval-key")
    r.add_argument("--preset", default="minimal-compact")
    r.add_argument("--only", default="")
    r.add_argument("--timeout-min", type=int, default=45)
    r.add_argument("--note", default="", help="what this run's harness config is")
    g = sub.add_parser("grade", help="grade one task workspace now (no agent)")
    g.add_argument("task_id")
    g.add_argument("--key", default=r"C:\tmp\agent-eval-key")
    p = sub.add_parser("report")
    p.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        Runner(a).serve()
    elif a.cmd == "grade":
        suite = {t["id"]: t for t in json.load(open(os.path.join(a.key, "suite.json"), encoding="utf-8"))}
        print(json.dumps(grade(suite[a.task_id]), indent=1))
    else:
        report(a.names)


if __name__ == "__main__":
    main()
