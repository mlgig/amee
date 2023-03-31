import numpy as np
from tqdm import tqdm

from experiments.base.exfiles import dataset_tab
from experiments.base.parallel import parallel as _parallel
from experiments.base.rng import reproducible_rng


def parallel(initial_max_workers, jobs, worker, logdir, job_to_logfile, job_to_msg):
    print(f"Initially running with {initial_max_workers} subprocesses.")
    print("Available commands:")
    print("  stop     Let all current subprocesses finish and then gracefully terminate.")
    print("  p <n>    Gracefully adjust the number of subprocesses to n.")

    _parallel(initial_max_workers, jobs, worker, logdir, job_to_logfile, job_to_msg, _command_listener)


def _command_listener(sig, command):
    command = command.strip().split(" ")
    if command[0] == "stop":
        tqdm.write("Will let all current subprocesses finish before terminating.")
        sig.stop = True
    elif command[0] == "p":
        if len(command) == 2 and command[1].isnumeric():
            tqdm.write(f"Will adjust the number of subprocesses to {command[1]}.")
            sig.max_workers = int(command[1])
        else:
            tqdm.write("Command 'p' expects exactly one integer parameter: p <n>")
    else:
        tqdm.write(f"Unknown command '{command[0]}'; available commands are: 'stop', 'p <n>'")


def resolve_dataset_names(archive_name, dataset_names):
    available_dataset_names = dataset_tab(archive_name)
    if dataset_names is None:
        return available_dataset_names
    else:
        nonexistent_dataset_names = set(dataset_names) - set(available_dataset_names)
        if len(nonexistent_dataset_names) != 0:
            print(f"Archive {archive_name} does not contain these explicitly specified datasets: "
                  ", ".join(nonexistent_dataset_names))
            raise ValueError
        else:
            return dataset_names


def find_specimen_idx(y_train, y_test, specimen_iter):
    # Get the label to randomly draw a specimen from.
    # Each iteration advances that label by 1.
    unique_labels = np.unique(y_train)
    y_specimen = unique_labels[specimen_iter % len(unique_labels)]

    # Find all indices of time series in the test set which have the label required for this iteration.
    possible_specimen_indices = np.where(y_test == y_specimen)[0]

    # If all possible specimen indices for this label have been exhausted during lower iterations, abort.
    n_previous_draws = specimen_iter // len(unique_labels)
    if n_previous_draws >= len(possible_specimen_indices):
        return None

    rng = reproducible_rng()
    # Throw away all specimen indices drawn during lower iterations.
    for i in range(n_previous_draws):
        draw = rng.randint(0, len(possible_specimen_indices))
        possible_specimen_indices = np.delete(possible_specimen_indices, draw)

    # Get the actual (reproducibly) random specimen from that set of time series.
    draw = rng.randint(0, len(possible_specimen_indices))
    return possible_specimen_indices[draw]
