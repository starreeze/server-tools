# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from __future__ import annotations
import os, traceback
from typing import Any, Callable, Iterable
from itertools import product
from io import TextIOWrapper
from multiprocessing import Lock, Process

# default file path
process_output = ".output_{}"
checkpoint_path = ".checkpoint_ir"


class IterateWrapper:
    "wrap some iterables to provide automatic resuming on interruption, no retrying and limited to sequence"

    def __init__(
        self,
        *data,
        mode="product",
        restart=False,
        bar=0,  # if -1 no bar at all
        total_items=None,
        convert_type=list,
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
        total_items = total_items if total_items is not None else len(self.data)  # type: ignore
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

    def __iter__(self):
        return self

    def __next__(self):
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(self.index))
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        return self.data[self.index]  # type: ignore


# def retry_fn(func: Callable, *args, retry=5, on_error="raise"):
#     for j in range(retry):
#         try:
#             return func(*args)
#         except KeyboardInterrupt:
#             traceback.print_exc()
#             exit(1)
#         except Exception as e:
#             if j == retry - 1:
#                 if on_error == "raise":
#                     print(f"All retry failed:")
#                     traceback.print_exc()
#                     return
#                 elif on_error == "continue":
#                     print(f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy.")
#             print(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")


def retry_dec(retry=5, on_error="raise"):
    "decorator for retrying a function on exception; on_error could be raise or continue"

    def decorator(func):
        def wrapper(*args, **kwargs):
            for j in range(retry):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    traceback.print_exc()
                    exit(1)
                except Exception as e:
                    if j == retry - 1:
                        if on_error == "raise":
                            print(f"All retry failed:")
                            traceback.print_exc()
                            return
                        elif on_error == "continue":
                            print(f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy.")
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
    func: Callable[[TextIOWrapper, Any], None],
    data: Iterable,
    total_items: int,
    process_idx: int,
    num_workers: int,
    iterator_mode: bool,
    lock,
    checkpoint_path,
    checkpoint=0,
    restart=False,
    retry=5,
    on_error="raise",
    bar=True,
    envs: dict[str, str] | None = None,
):
    if envs is not None:
        for k, v in envs.items():
            os.environ[k] = v

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
            next(data)  # type: ignore

    with open(process_output.format(process_idx), "w" if restart else "a") as output:
        for i in range_to_process:
            if iterator_mode:
                try:
                    item = next(data)  # type: ignore
                except StopIteration:
                    break
            else:
                item = data[i]  # type: ignore
            retry_func = retry_dec(retry, on_error)(func)
            retry_func(output, item)
            write_ckpt(checkpoint_path, i + 1, process_idx, lock)


def merge_files(input_paths: list[str], output_stream: TextIOWrapper):
    for path in input_paths:
        with open(path, "r") as f:
            output_stream.write(f.read())
        os.remove(path)


def iterate_wrapper(
    func: Callable[[TextIOWrapper, Any], None],
    data: Iterable,
    output_stream: str | TextIOWrapper,
    restart=False,
    retry=5,
    on_error="raise",
    num_workers=1,
    bar=True,
    total_items=None,  # need to provide when data is not a sequence
    envs: list[dict[str, str]] = [],  # additional env for each workers
) -> None:
    """
    Need hand-crafted function but support retrying, multiprocessing and iterable.
    Only support io stream to save the processed data.
    Callbacks can be implemented by setting envs, e.g. PROC_ID or GPU_ID.
    """
    # init vars
    if num_workers < 1 or len(envs) and len(envs) != num_workers:
        raise ValueError("num_workers must be a positive integer and envs must be a list of length num_workers")
    if total_items is None:
        total_items = len(data)  # type: ignore
    try:
        data[0]  # type: ignore
        iterator_mode = False
    except TypeError:
        data = iter(data)
        iterator_mode = True

    # load checkpoint
    checkpoint = load_ckpt(checkpoint_path, restart)
    if checkpoint is None:
        checkpoint = [0] * num_workers
        with open(checkpoint_path, "w") as f:
            f.write("\n".join(map(str, checkpoint)))
    elif len(checkpoint) != num_workers:
        raise ValueError(f"checkpoint length {len(checkpoint)} does not match num_workers {num_workers}!")

    # get multiprocessing results
    lock = Lock()
    get_job_args = lambda i: (
        func,
        data,
        total_items,
        i,
        num_workers,
        iterator_mode,
        lock,
        checkpoint_path,
        checkpoint[i],
        restart,
        retry,
        on_error,
        bar,
        os.environ | (envs[i] if len(envs) else {}),
    )
    pool = [Process(target=process_job, args=get_job_args(i)) for i in range(num_workers)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
        p.close()

    # merge multiple files
    if isinstance(output_stream, str):
        f = open(output_stream, "w" if restart else "a")
    else:
        f = output_stream
    merge_files([process_output.format(i) for i in range(num_workers)], f)

    # remove checkpoint file on complete
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


# testing
tmp = True


def perform_operation(_, item):
    from time import sleep

    global tmp
    sleep(1)
    if item == 0:
        if tmp:
            tmp = False
            raise ValueError("here")


def test_fn():
    data = list(range(10))
    iterate_wrapper(perform_operation, data, "output.txt", num_workers=3)


def test_wrapper():
    from time import sleep

    for i in IterateWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    test_fn()
