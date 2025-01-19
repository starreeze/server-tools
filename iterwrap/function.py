import os
from functools import partial
from multiprocessing import Lock, Process, Queue, synchronize
from typing import (
    IO,
    BinaryIO,
    Callable,
    Concatenate,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TextIO,
    Union,
    cast,
)

from tqdm import tqdm

from .utils import (
    DataType,
    ParamTypes,
    ReturnType,
    ckpt_tmpl,
    load_ckpt,
    logger,
    merge_files,
    output_tmpl,
    retry_dec,
    write_ckpt,
)


def _process_job(
    func: Callable[[IO, DataType], ReturnType],
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
) -> Sequence[ReturnType]:
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
        range_to_process = tqdm(
            range_to_process,
            initial=checkpoint - start_pos,
            total=end_pos - start_pos,
            position=process_idx,
            desc=f"proc {process_idx}",
        )

    if iterator_mode:
        logger.info(f"{process_idx}: skipping data until {checkpoint}")
        assert isinstance(data, Iterator)
        for i in range_checkpointed:
            next(data)
    open_flags = ("w" if restart else "a") + ("b" if output_type == "binary" else "")
    output = open(output_tmpl.format(name=run_name, id=process_idx), open_flags) if output_type != "none" else None

    logger.debug(f"{process_idx}: processing data from {checkpoint} to {end_pos}")

    returns = []
    for i in range_to_process:
        logger.debug(f"{process_idx}: processing data {i}")
        if iterator_mode:
            assert isinstance(data, Iterator)
            try:
                item = next(data)
            except StopIteration:
                break
        else:
            assert isinstance(data, Sequence)
            item = data[i]
        returns.append(func(output, item))  # type: ignore
        if output is not None and flush:
            output.flush()
        write_ckpt(ckpt_tmpl.format(name=run_name), i + 1 - start_pos, process_idx, lock)
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


def iterate_wrapper(
    func: Union[
        # Callable[Concatenate[IO, DataType, dict[str, Any], ParamTypes], ReturnType],
        Callable[Concatenate[IO, DataType, ParamTypes], ReturnType],
        Callable[Concatenate[DataType, ParamTypes], ReturnType],
    ],
    data: Iterable[DataType],
    output: str | IO | None = None,
    restart=False,
    retry=5,
    wait=1,
    on_error: Literal["raise", "continue"] = "raise",
    num_workers=1,
    bar=True,
    flush=True,
    total_items: int | None = None,
    run_name=__name__,
    envs: list[dict[str, str]] = [],
    # vars_factory: Callable[[], dict[str, Any]] = lambda: {},
    *args: ParamTypes.args,
    **kwargs: ParamTypes.kwargs,
) -> Sequence[ReturnType] | None:
    """Wrapper on a processor (func) and iterable (data) to support multiprocessing, retrying and automatic resuming.

    Args:
        func: The processor function. It should accept the following argument patterns: data item only; output stream, data item. Additional args (*args and **kwargs) can be added in func, which should be passed to the wrapper. Within func, the output stream can be used to save data in real time.
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
        logger.warning(
            f"num_workers {num_workers} is greater than total_items {total_items}, "
            "setting num_workers to total_items."
        )
        num_workers = total_items

    # load checkpoint
    checkpoint_path = ckpt_tmpl.format(name=run_name)
    checkpoint = load_ckpt(checkpoint_path, restart)
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
            matched_func = lambda io, data_item: partial_func(data_item)  # type: ignore
        case 2:  # io, data_item
            matched_func = lambda io, data_item: partial_func(io, data_item)  # type: ignore
        case _:
            raise ValueError(
                f"func must accept 1 or 2 arguments, got {func.__code__.co_argcount - len(args) - len(kwargs)}"
            )
    retry_func = cast(Callable[[IO, DataType], ReturnType], retry_dec(retry, wait, on_error)(matched_func))

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
    merge_files([output_tmpl.format(name=run_name, id=i) for i in range(num_workers)], f)
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
        logger.warning(
            "The run is once interrupted and the return value is incomplete. "
            "You can safely ignore this if you save your results in real-time."
        )
    return None
