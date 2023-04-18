import contextlib
from datetime import datetime as dt
import os

from torch.profiler import profile, ProfilerActivity, schedule

def pytorch_profile(name, fake=True, kill=True, warmup=0):
    if fake:
        return contextlib.suppress()
    print(f'starting profiler: {name}')
    def trace_handler(prof_):
        os.makedirs('profiling_outputs', exist_ok=True)
        fn = f'profiling_outputs/{name}_profile {dt.strftime(dt.now(), "%c").replace(":", "_")}.json'
        prof_.export_chrome_trace(fn)
        with open(fn, 'r') as f:
            s = f.read()
        with open(fn, 'w') as f:
            f.write(s.replace('N/A', '95054'))  # avoid error in chrome trace viewer
        if kill:
            raise Exception('killed by profiling')
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True, schedule=schedule(wait=0, warmup=warmup, active=1),
        on_trace_ready=trace_handler
    )
    return prof