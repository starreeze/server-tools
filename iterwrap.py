# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from __future__ import annotations

import json
import logging
import os
import traceback
from functools import partial, wraps
from glob import glob
from itertools import product
from multiprocessing import Lock, Process, Queue, synchronize
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    Sequence,
    TextIO,
    TypeVar,
    cast,
)

from tqdm import tqdm

# package info
__version__ = "0.1.8"
__author__ = "Starreeze"
__license__ = "GPLv3"
__url__ = "https://github.com/starreeze/server-tools"

# default file path
output_tmpl = "{name}_p{id}.output"
ckpt_tmpl = "{name}.ckpt"

# type
DataType = TypeVar("DataType")  # the data type of the items to be processed
ReturnType = TypeVar("ReturnType")  # the data type of the items to be returned
ParamTypes = ParamSpec("ParamTypes")  # the additional parameters of the processor function


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_tqdm_logger(name=__name__, fmt="%(asctime)s - %(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


_logger = setup_tqdm_logger()


def check_unfinished(run_name: str):
    "To check if the run is unfinished, according to whether the checkpoint file and the cache file exists"
    ckpt = ckpt_tmpl.format(name=run_name)
    if os.path.exists(ckpt):
        num_cache = len(glob(output_tmpl.format(name=run_name, id="*")))
        num_ckpt = len(open(ckpt, "r").readlines())
        if num_cache == num_ckpt:
            return True
        _logger.warning(
            f"unmatched: {num_cache} unfinished files vs {num_ckpt} checkpoints, restart from the beginning"
        )
    return False


class IterateWrapper(Generic[DataType]):
    def __init__(
        self,
        *data: Iterable[DataType],
        mode: Literal["product", "zip"] = "product",
        restart=False,
        bar=0,
        total_items: int | None = None,
        convert_type=list,
        run_name=__name__,
    ):
        """wrap some iterables to provide automatic resuming on interruption, no retrying and limited to sequence

        Args:
            data: iterables to be wrapped
            mode: how to combine iterables. 'product' means Cartesian product, 'zip' means zip()
            restart: whether to restart from the beginning
            bar: the position of the progress bar. -1 means no bar
            total_items: total items to be iterated
            convert_type: convert the data to this type
            run_name: name of the run to identify the checkpoint and output files
        """
        if mode == "product":
            self.data: Sequence[DataType] = convert_type(product(*data))
        elif mode == "zip":
            self.data = convert_type(zip(*data))
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
            self.wrapped_range = tqdm(
                range(checkpoint, total_items),
                initial=checkpoint,
                total=total_items,
                position=bar,
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


def retry_dec(retry=5, on_error: Literal["raise", "continue"] = "raise"):
    "decorator for retrying a function on exception; on_error could be raise or continue"

    def decorator(func):
        def wrapper(*args, **kwargs):
            if retry <= 1:
                return func(*args, **kwargs)
            for j in range(retry):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt as e:
                    _logger.error(traceback.format_exc())
                    raise e
                except BaseException as e:
                    if j == retry - 1:
                        if on_error == "raise":
                            _logger.error("All retry failed:")
                            _logger.info(traceback.format_exc())
                            raise e
                        elif on_error == "continue":
                            _logger.warning(
                                f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy."
                            )
                            return
                    _logger.warning(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")
                    _logger.debug(traceback.format_exc())

        return wrapper

    return decorator


def _load_ckpt(path: str, restart: bool) -> list[int] | None:
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


def _write_ckpt(path: str, checkpoint: int, process_idx: int, lock: synchronize.Lock):
    with lock:
        with open(path, "r") as f:
            checkpoints = f.read().splitlines()
        checkpoints[process_idx] = str(checkpoint)
        with open(path, "w") as f:
            f.write("\n".join(checkpoints))


def _process_job(
    func: Callable[[IO, DataType, dict[str, Any]], ReturnType],
    data: Iterator[DataType] | Sequence[DataType],
    output_type: Literal["text", "binary", "none"],
    total_items: int,
    process_idx: int,
    num_workers: int,
    iterator_mode: bool,
    lock: synchronize.Lock,
    run_name,
    checkpoint,
    restart,
    bar,
    flush,
    envs: dict[str, str] | None,
    vars_factory: Callable[[], dict[str, Any]],
) -> Sequence[ReturnType]:
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
        range_to_process = tqdm(
            range_to_process,
            initial=checkpoint - start_pos,
            total=end_pos - start_pos,
            position=process_idx,
            desc=f"proc {process_idx}",
        )

    if iterator_mode:
        _logger.info(f"{process_idx}: skipping data until {checkpoint}")
        assert isinstance(data, Iterator)
        for i in range_checkpointed:
            next(data)
    open_flags = ("w" if restart else "a") + ("b" if output_type == "binary" else "")
    output = open(output_tmpl.format(name=run_name, id=process_idx), open_flags) if output_type != "none" else None

    _logger.debug(f"{process_idx}: processing data from {checkpoint} to {end_pos}")

    returns = []
    for i in range_to_process:
        _logger.debug(f"{process_idx}: processing data {i}")
        if iterator_mode:
            assert isinstance(data, Iterator)
            try:
                item = next(data)
            except StopIteration:
                break
        else:
            assert isinstance(data, Sequence)
            item = data[i]
        returns.append(func(output, item, vars))  # type: ignore
        if output is not None and flush:
            output.flush()
        _write_ckpt(ckpt_tmpl.format(name=run_name), i + 1 - start_pos, process_idx, lock)
    if output is not None:
        output.close()
    return returns


class WorkerProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.queue = Queue()  # Add queue for communication

    def run(self):
        returns = _process_job(*self.args, **self.kwargs)
        self.queue.put(returns)  # Put results in queue

    def get_returns(self):
        return self.queue.get()  # Get results from queue


def _merge_files(input_paths: list[str], output_stream: IO | None):
    for path in input_paths:
        if output_stream is not None:
            with open(path, "r") as f:
                output_stream.write(f.read())
        if os.path.exists(path):
            os.remove(path)


def iterate_wrapper(
    func: (
        Callable[Concatenate[DataType, ParamTypes], ReturnType]
        | Callable[Concatenate[IO, DataType, ParamTypes], ReturnType]
        | Callable[Concatenate[IO, DataType, dict[str, Any], ParamTypes], ReturnType]
    ),
    data: Iterable[DataType],
    output: str | IO | None = None,
    restart=False,
    retry=5,
    on_error: Literal["raise", "continue"] = "raise",
    num_workers=1,
    bar=True,
    flush=True,
    total_items: int | None = None,
    run_name=__name__,
    envs: list[dict[str, str]] = [],
    vars_factory: Callable[[], dict[str, Any]] = lambda: {},
    *args: ParamTypes.args,
    **kwargs: ParamTypes.kwargs,
) -> Sequence[ReturnType] | None:
    """Wrapper on a processor (func) and iterable (data) to support multiprocessing, retrying and automatic resuming.

    Args:
        func: The processor function. It should accept three or more arguments: output stream, data item, vars, and additional args (*args and **kwargs, which should be passed to the wrapper). Within func, the output stream can be used to save data in real time.
        data: The data to be processed. It can be an iterable or a sequence. In each iteration, the data item in data will be passed to func.
        output: The output stream. It can be a file path, a file object or None. If None, no output will be written.
        restart: Whether to restart from the beginning.
        retry: The number of retries for processing each data item.
        on_error: The action to take when an exception is raised in func.
        num_workers: The number of workers to use. If set to 1, the processor will be run in the main process.
        bar: Whether to show a progress bar (package tqdm required).
        flush: Whether to flush the output stream after each data item is processed.
        total_items: The total number of items in data. It is required when data is not a sequence.
        run_name: The name of the run. It is used to construct the checkpoint file path.
        envs: Additional environment variables for each worker. This will be set before spawning new processes.
        vars_factory: A function that returns a dictionary of variables to be passed to func. The factory will be called after each process is spawned and before entering the loop. For plain vars, one can simply use closure or functools.partial to pass into func.
        *args: Additional positional arguments to be passed to func.
        **kwargs: Additional keyword arguments to be passed to func.

    Returns:
        A list of return values from func.
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

    if num_workers > total_items:
        _logger.warning(
            f"num_workers {num_workers} is greater than total_items {total_items}, "
            "setting num_workers to total_items."
        )
        num_workers = total_items

    # load checkpoint
    checkpoint_path = ckpt_tmpl.format(name=run_name)
    checkpoint = _load_ckpt(checkpoint_path, restart)
    if checkpoint is None:
        checkpoint = [0] * num_workers
        with open(checkpoint_path, "w") as f:
            f.write("\n".join(map(str, checkpoint)))
        return_flag = True
    else:
        if len(checkpoint) != num_workers:
            raise ValueError(f"checkpoint length {len(checkpoint)} does not match num_workers {num_workers}!")
        max_split = (total_items + num_workers - 1) // num_workers
        if not all(map(lambda x: 0 <= x < max_split, checkpoint)):
            raise ValueError(f"checkpoint must be a list of non-negative integers less than max_split {max_split}")
        return_flag = False

    # get multiprocessing results
    lock = Lock()
    if isinstance(output, str) or isinstance(output, TextIO):
        output_type = "text"
    elif isinstance(output, BinaryIO):
        output_type = "binary"
    else:
        if output is not None:
            raise ValueError("output must be a string, a file-like object or None")
        output_type = "none"

    partial_func = partial(func, *args, **kwargs)
    match func.__code__.co_argcount - len(args) - len(kwargs):
        case 1:  # data_item only
            matched_func = lambda io, data_item, vars: partial_func(data_item)  # type: ignore
        case 2:  # io, data_item
            matched_func = lambda io, data_item, vars: partial_func(io, data_item)  # type: ignore
        case 3:  # io, data_item, vars
            matched_func = partial_func
        case _:
            raise ValueError(f"func must accept 1, 2 or 3 arguments, got {func.__code__.co_argcount}")
    retry_func = cast(Callable[[IO, DataType, dict[str, Any]], ReturnType], retry_dec(retry, on_error)(matched_func))

    def _get_job_args(i: int):
        return (
            retry_func,
            data,
            output_type,
            total_items,
            i,
            num_workers,
            iterator_mode,
            lock,
            run_name,
            checkpoint[i],
            restart,
            bar,
            flush,
            os.environ | (envs[i] if len(envs) else {}),
            vars_factory,
        )

    if num_workers > 1:
        returns = []
        pool = [WorkerProcess(*_get_job_args(i)) for i in range(num_workers)]
        for p in pool:
            p.start()
        for p in pool:
            p.join()
            returns.extend(p.get_returns())  # Get results from queue
            p.close()
    else:
        returns = _process_job(*_get_job_args(0))

    # merge multiple files
    if isinstance(output, str):
        dir = os.path.dirname(output)
        if dir and not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        f = open(output, "w" if restart else "a")
    elif isinstance(output, IO):
        f = output
    else:
        f = None
    _merge_files([output_tmpl.format(name=run_name, id=i) for i in range(num_workers)], f)
    if f is not None:
        f.close()

    # remove checkpoint file on complete
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass

    if return_flag:
        return returns
    if returns[0] is not None:
        _logger.warning(
            "The run is once interrupted and the return value is incomplete. "
            "You can safely ignore this if you save your results in real-time."
        )
    return None


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
                _logger.info(f"Loading cached file {file_path} from disk...")
                with open(file_path, "r") as f:
                    return json.load(f)

        return wrapper

    return decorator


# testing
def _perform_operation(item: int, sleep_time: float):
    from time import sleep

    global _tmp
    sleep(sleep_time)
    if item == 0:
        if _tmp:
            _tmp = False
            raise ValueError("here")
    return item * item


def _test_fn():
    data = list(range(10))
    num_workers = 3
    returns = iterate_wrapper(
        _perform_operation,
        data,
        "output.txt",
        num_workers=num_workers,
        sleep_time=1,
    )
    print(returns)


def _test_wrapper():
    from time import sleep

    for i in IterateWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    _tmp = True
    _test_fn()
