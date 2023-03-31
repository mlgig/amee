import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from select import select

import numpy as np
from tqdm import tqdm


class Skip(Exception):
    pass


@dataclass
class SignalContainer:
    max_workers: int
    stop: bool


def parallel(max_workers, jobs, worker, logdir=None, job_to_logfile=None, job_to_msg=None, handle_stdin=None):
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)

    jobs = list(jobs)  # collect all jobs
    processes = []

    sig = SignalContainer(max_workers, stop=False)

    with tqdm(total=len(jobs)) as pbar:
        while len(jobs) != 0 or len(processes) != 0:
            time.sleep(1)

            # If there is input on stdin, handle it
            if handle_stdin is not None and select([sys.stdin, ], [], [], 0)[0]:
                handle_stdin(sig, sys.stdin.readline())

            # Remove all terminated processes from the list and update progress bar
            terminated_indices = [idx for idx, (job, proc) in enumerate(processes) if not proc.is_alive()]
            for idx in sorted(terminated_indices, reverse=True):  # reverse order to not mess up following indices
                job, proc = processes[idx]
                if proc.exitcode != 2:  # 2 stands for skipped
                    tqdm.write(("Done: " if proc.exitcode == 0 else "Failed: ") + job_to_msg(**job))
                pbar.update()
                # Delete
                del processes[idx]

            # If we are requested to stop, forget all future jobs
            if sig.stop:
                pbar.total -= len(jobs)
                del jobs[:]

            # If there are open worker slots and jobs left, create new processes accordingly
            while len(processes) < sig.max_workers and len(jobs) != 0:
                job = jobs.pop(0)
                proc = Process(name="Job-" + "-".join(map(str, job.values())),
                               target=_worker_wrapper, args=(job, worker, logdir, job_to_logfile))
                proc.start()
                processes.append((job, proc))


def _worker_wrapper(job, worker, logdir, job_to_logfile):
    # By default, the child worker process inherits its RNG seed from the parent process.
    # That is bad if we perform a (partially random) computation multiple times and want to compare the results.
    # To fix it, we seed each child process with a new number from the OS pool of random numbers.
    np.random.seed()

    if logdir is None or job_to_logfile is None:
        worker(**job)
    else:
        # Reroute native print output to /dev/null so it doesn't clutter the main terminal
        with open(os.devnull, "w") as to:
            os.dup2(to.fileno(), sys.stdout.fileno())
            os.dup2(to.fileno(), sys.stderr.fileno())

        logfile = os.path.join(logdir, job_to_logfile(**job))
        if os.path.exists(logfile):
            os.remove(logfile)

        logging.root.handlers = []  # fix stupid tampering of global logging state by abseil
        logging.basicConfig(filename=logfile, level=logging.INFO, format="%(asctime)s %(levelname)8s - %(message)s")

        sys.stdout.write = lambda msg: _log_if_not_empty(logging.getLogger().info, msg)
        sys.stderr.write = lambda msg: _log_if_not_empty(logging.getLogger().error, msg)

        try:
            print(f"Running in process with PID {os.getpid()}")
            worker(**job)
            sys.exit(0)
        except Skip:
            sys.exit(2)
        except Exception:  # don't catch anything more general than Exception to not block the SystemExit exception
            traceback.print_exc()
            sys.exit(1)


def _log_if_not_empty(logging_func, msg):
    if msg.strip() != "":
        logging_func(msg.strip())
