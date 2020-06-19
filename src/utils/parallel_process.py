from tqdm import tqdm
import subprocess


def parallel_process(f, params, parallelism):
    procs = []
    try:
        with tqdm(total=len(params)) as pbar:
            i = 0
            while len(params) > 0 or len(procs) > 0:
                if len(procs) < parallelism and len(params) > 0:
                    param = params.pop()
                    procs.append(f(*param))
                elif len(procs) > 0:
                    try:
                        procs[i].wait(1)
                        assert procs[i].returncode == 0
                        pbar.update(1)
                        pbar.refresh()
                        procs.pop(i)
                    except subprocess.TimeoutExpired:
                        pass
                    i = (i + 1) % max(len(procs), 1)
    finally:
        for p in procs:
            p.terminate()
