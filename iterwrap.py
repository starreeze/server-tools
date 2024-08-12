# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from __future__ import annotations
import os
import traceback
import json
from typing import Any, Callable, Iterable, Iterator, TypeVar, TextIO, Sequence
from glob import glob
from itertools import product
from functools import wraps
from multiprocessing import Lock, Process

# default file path
output_tmpl = "{name}_p{id}.output"
ckpt_tmpl = "{name}.ckpt"

DataType = TypeVar("DataType")


def check_unfinished(run_name: str):
    ckpt = ckpt_tmpl.format(name=run_name)
    if os.path.exists(ckpt):
        return len(glob(output_tmpl.format(name=run_name, id="*"))) == len(open(ckpt, "r").readlines())
    return False


class IterateWrapper:
    "wrap some iterables to provide automatic resuming on interruption, no retrying and limited to sequence"

    def __init__(
        self,
        *data: Any,
        mode="product",
        restart=False,
        bar=0,  # if -1 no bar at all
        total_items: int | None = None,
        convert_type=list,
        run_name="iterwrap",
    ):
        if len(data) == 1:
            if convert_type is not None:
                self.data = convert_type(data[0])
            else:
                self.data = data[0]
        elif mode == "product":
            self.data = list(product(*data))
        elif mode == "zip":
            self.data = list(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")
        total_items = total_items if total_items is not None else len(self.data)
        checkpoint_path = ckpt_tmpl.format(name=run_name)
        if restart:
            os.remove(checkpoint_path)
        try:
            with open(checkpoint_path, "r") as checkpoint_file:
                checkpoint = int(checkpoint_file.read().strip())
        except FileNotFoundError:
            checkpoint = 0
        if bar >= 0:
            from tqdm import tqdm

            self.wrapped_range = tqdm(
                range(checkpoint, total_items), initial=checkpoint, total=total_items, position=bar
            )
        else:
            self.wrapped_range = range(checkpoint, total_items)
        self.wrapped_iter = iter(self.wrapped_range)
        self.index = 0
        self.checkpoint_path = checkpoint_path

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(self.index))
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(self.checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        return self.data[self.index]


def retry_dec(retry=5, on_error="raise"):
    "decorator for retrying a function on exception; on_error could be raise or continue"

    def decorator(func):
        def wrapper(*args, **kwargs):
            if retry <= 1:
                return func(*args, **kwargs)
            for j in range(retry):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt as e:
                    traceback.print_exc()
                    raise e
                except BaseException as e:
                    if j == retry - 1:
                        if on_error == "raise":
                            print("All retry failed:")
                            traceback.print_exc()
                            raise e
                        elif on_error == "continue":
                            print(f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy.")
                            return
                    print(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")

        return wrapper

    return decorator


def load_ckpt(path: str, restart: bool) -> list[int] | None:
    checkpoint = None
    if restart:
        if os.path.exists(path):
            os.remove(path)
    try:
        with open(path, "r") as checkpoint_file:
            checkpoint = list(map(int, checkpoint_file.read().splitlines()))
    except FileNotFoundError:
        pass
    return checkpoint


def write_ckpt(path: str, checkpoint: int, process_idx: int, lock):
    with lock:
        with open(path, "r") as f:
            checkpoints = f.read().splitlines()
        checkpoints[process_idx] = str(checkpoint)
        with open(path, "w") as f:
            f.write("\n".join(checkpoints))


def process_job(
    func: Callable[[TextIO, DataType, dict[str, Any]], None],
    data: Iterator[DataType],
    total_items: int,
    process_idx: int,
    num_workers: int,
    iterator_mode: bool,
    lock,
    run_name,
    checkpoint,
    restart,
    retry,
    on_error,
    bar,
    envs: dict[str, str] | None,
    vars_factory: Callable[[], dict[str, Any]],
):
    if envs is not None:
        for k, v in envs.items():
            os.environ[k] = v
    vars = vars_factory()

    chunk_items = (total_items + num_workers - 1) // num_workers
    start_pos = process_idx * chunk_items
    end_pos = min(start_pos + chunk_items, total_items)
    checkpoint += start_pos

    range_to_process = range(checkpoint, end_pos)
    range_checkpointed = range(start_pos, checkpoint)
    if bar:
        from tqdm import tqdm

        range_to_process = tqdm(
            range_to_process,
            initial=checkpoint - start_pos,
            total=end_pos - start_pos,
            position=process_idx,
            desc=f"proc {process_idx}",
        )

    if iterator_mode:
        print(f"{process_idx}: skipping data until {checkpoint}")
        for i in range_checkpointed:
            next(data)

    retry_func = retry_dec(retry, on_error)(func)
    with open(output_tmpl.format(name=run_name, id=process_idx), "w" if restart else "a") as output:
        for i in range_to_process:
            if iterator_mode:
                try:
                    item = next(data)
                except StopIteration:
                    break
            else:
                assert isinstance(data, Sequence)
                item = data[i]
            retry_func(output, item, vars)
            write_ckpt(ckpt_tmpl.format(name=run_name), i + 1, process_idx, lock)


def merge_files(input_paths: list[str], output_stream: TextIO):
    for path in input_paths:
        with open(path, "r") as f:
            output_stream.write(f.read())
        os.remove(path)


def iterate_wrapper(
    func: Callable[[TextIO, DataType, dict[str, Any]], None],
    data: Iterable[DataType],
    output: str | TextIO,
    restart=False,
    retry=5,
    on_error="raise",
    num_workers=1,
    bar=True,
    total_items=None,  # need to provide when data is not a sequence
    run_name="iterwrap",
    envs: list[dict[str, str]] = [],  # additional env for each workers
    # construct vars that will be passed to func, after each process is spawned and before entering the loop
    # for plain vars, one can simply use closure or functools.partial to pass into func
    vars_factory: Callable[[], dict[str, Any]] = lambda: {},
) -> None:
    """
    Need hand-crafted function but support retrying, multiprocessing and iterable.
    Only support io stream to save the processed data.
    Callbacks can be implemented by setting envs, e.g. PROC_ID or GPU_ID.
    """
    # init vars
    if num_workers < 1 or len(envs) and len(envs) != num_workers:
        raise ValueError("num_workers must be a positive integer and envs must be a list of length num_workers")
    if isinstance(data, Sequence):
        iterator_mode = False
        if total_items is not None:
            assert total_items == len(data)
        else:
            total_items = len(data)
    else:
        iterator_mode = True
        data = iter(data)
        assert total_items is not None, "total_items must be provided when data is not a sequence"

    # load checkpoint
    checkpoint_path = ckpt_tmpl.format(name=run_name)
    checkpoint = load_ckpt(checkpoint_path, restart)
    if checkpoint is None:
        checkpoint = [0] * num_workers
        with open(checkpoint_path, "w") as f:
            f.write("\n".join(map(str, checkpoint)))
    elif len(checkpoint) != num_workers:
        raise ValueError(f"checkpoint length {len(checkpoint)} does not match num_workers {num_workers}!")

    # get multiprocessing results
    lock = Lock()

    def get_job_args(i: int):
        return (
            func,
            data,
            total_items,
            i,
            num_workers,
            iterator_mode,
            lock,
            run_name,
            checkpoint[i],
            restart,
            retry,
            on_error,
            bar,
            os.environ | (envs[i] if len(envs) else {}),
            vars_factory,
        )

    pool = [Process(target=process_job, args=get_job_args(i)) for i in range(num_workers)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
        p.close()

    # merge multiple files
    if isinstance(output, str):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        f = open(output, "w" if restart else "a")
    else:
        f = output
    merge_files([output_tmpl.format(name=run_name, id=i) for i in range(num_workers)], f)

    # remove checkpoint file on complete
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


def bind_cache_json(file_path_factory: Callable[[], str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            file_path = file_path_factory()
            if not os.path.exists(file_path):
                result = func(*args, **kwargs)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    json.dump(result, f)
                return result
            else:
                print(f"Loading cached file {file_path} from disk...")
                with open(file_path, "r") as f:
                    return json.load(f)

        return wrapper

    return decorator


# testing
tmp = True


def perform_operation(_, item: int, vars):
    from time import sleep

    global tmp
    print(vars)
    sleep(1)
    if item == 0:
        if tmp:
            tmp = False
            raise ValueError("here")


def test_fn():
    data = list(range(10))
    l = [0, 1]  # noqa: E741
    iterate_wrapper(
        perform_operation,
        data,
        "output.txt",
        num_workers=3,
        envs=[{"id": str(i)} for i in range(3)],
        vars_factory=lambda: {"a": l + [os.environ["id"]]},
    )


def test_wrapper():
    from time import sleep

    for i in IterateWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    test_fn()
