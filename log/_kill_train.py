import os, signal, subprocess
out = subprocess.getoutput("ps -eo pid,cmd")
for line in out.splitlines():
    if any(x in line for x in ("train-glm52-2node", "scripts/train.py", "torchrun")) and "_kill_train" not in line:
        try:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)
            print("killed", pid)
        except Exception as e:
            print("skip", e)
print("done")
