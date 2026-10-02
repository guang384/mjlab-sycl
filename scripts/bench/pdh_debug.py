"""One controlled test: launch an in-process GPU load thread, then sample
typeperf's GPU Engine counters and check whether Level-Zero compute shows
up at all. Prints the hog pid's instances and their values."""
import subprocess, csv, sys, time, threading

result = {}

def hog():
    import torch
    a = torch.randn(32 * 1024 * 1024, device="xpu")  # 128 MB
    b = torch.empty_like(a)
    end = time.time() + 12
    n = 0
    while time.time() < end:
        b.copy_(a); n += 1
    result["bursts"] = n

t = threading.Thread(target=hog)
t.start()
time.sleep(6)  # let the load spin up

out = subprocess.run(
    ["typeperf", "\\GPU Engine(*)\\Utilization Percentage", "-sc", "2", "-si", "1"],
    capture_output=True, timeout=30).stdout.decode("utf-8", errors="replace")
t.join()
rows = [r for r in csv.reader(out.splitlines()) if len(r) > 5]
print("rows:", [len(r) for r in rows])
header, values = rows[0], rows[-1]
per_pid = {}
for h, v in zip(header[1:], values[1:]):
    if "pid_" not in h:
        continue
    pid = h.split("pid_")[1].split("_")[0]
    try:
        val = float(v)
    except ValueError:
        continue
    if val > 0:
        per_pid.setdefault(pid, 0.0)
        per_pid[pid] += val
print("hog bursts:", result.get("bursts"))
print("pids with nonzero utilization:", {k: round(v, 1) for k, v in sorted(per_pid.items(), key=lambda kv: -kv[1])[:8]})
# also show what instances exist for the busiest pid in the header
hots = max(per_pid.items(), key=lambda kv: kv[1]) if per_pid else None
if hots:
    inst = [h for h in header if f"pid_{hots[0]}_" in h][:12]
    print("instances of hottest pid:", [i.split("eng_")[1][:40] for i in inst])
