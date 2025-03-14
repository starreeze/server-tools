import json
import os
from functools import partial
from typing import IO, Callable, Concatenate, Iterable, Iterator, Literal, Sequence, Union, cast

from multiprocess import Lock, Process, synchronize  # type: ignore
from tqdm import tqdm

from .utils import (
    DataType,
    ParamTypes,
    ReturnType,
    clean_up,
    default_tmp_dir,
    get_checkpoint_path,
    get_output_path,
    get_partial_argcount,
    load_ckpt,
    logger,
    merge_files,
    retry_dec,
    write_ckpt,
)


def _process_job(
    func: Callable[[IO, DataType], ReturnType],
    data: Iterator[DataType] | Sequence[DataType],
    output_type: Literal["text", "binary"],
    total_items: int,
    process_idx: int,
    num_workers: int,
    iterator_mode: bool,
    lock: synchronize.Lock,
    run_name,
    tmp_dir: str,
    checkpoint,
    restart,
    bar,
    flush,
    envs: dict[str, str] | None,
) -> Sequence[ReturnType]:
    if envs is not None:
        os.environ.update(envs)

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
    output = open(get_output_path(run_name, process_idx, tmp_dir), open_flags)

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
        if flush:
            output.flush()
            write_ckpt(get_checkpoint_path(run_name, tmp_dir), i + 1 - start_pos, process_idx, lock)
    output.close()
    return returns


class WorkerProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def run(self):
        _process_job(*self.args, **self.kwargs)

    def get_returns(
        self,
        output_type: Literal["text", "binary"],
        binary_sep: bytes,
        binary_load_fn: Callable[[bytes], list[ReturnType]] | None,
    ) -> list[ReturnType]:
        # Read results from the output file created by _process_job
        results_path = get_output_path(self.args[8], self.args[4], self.args[9])
        if output_type == "text":
            results = []
            with open(results_path) as f:
                for line in f:
                    results.append(json.loads(line))
            return results
        with open(results_path, "rb") as f:
            content = f.read()
        if binary_load_fn is None:
            return content.split(binary_sep)  # type: ignore
        else:
            return binary_load_fn(content)


def resume_progress(run_name: str, restart: bool, num_workers: int, total_items: int, tmp_dir: str):
    checkpoint_path = get_checkpoint_path(run_name, tmp_dir)
    checkpoint = load_ckpt(checkpoint_path, restart)
    if checkpoint is None:
        checkpoint = [0] * num_workers
        with open(checkpoint_path, "w") as f:
            f.write("\n".join(map(str, checkpoint)))
    else:
        if len(checkpoint) != num_workers:
            raise ValueError(f"checkpoint length {len(checkpoint)} does not match num_workers {num_workers}!")
        max_split = (total_items + num_workers - 1) // num_workers
        if not all(map(lambda x: 0 <= x <= max_split, checkpoint)):
            raise ValueError(
                f"checkpoint must be a list of non-negative integers less than or equal to max_split {max_split}"
            )
        logger.warning(
            f"Resuming from checkpoint. This is expected if you are resuming from a previous interruption, "
            "but NOT expected if you are processing fresh data, "
            "in which case please specify tmp_dir when calling iterate_wrapper."
        )
        logger.warning(
            f"You can also choose to delete the checkpoint file `{checkpoint_path}` "
            "or specify restart=True when calling iterate_wrapper to restart from the beginning."
        )
    return checkpoint


def wrap_func(
    func: Union[
        Callable[Concatenate[DataType, ParamTypes], ReturnType],
        Callable[Concatenate[IO, DataType, ParamTypes], ReturnType],
    ],
    output_type: Literal["text", "binary"],
    binary_sep: bytes,
    *args: ParamTypes.args,
    **kwargs: ParamTypes.kwargs,
) -> Callable[[IO, DataType], ReturnType]:
    """
    Wrap the user-defined function by inserting additional arguments and handling the output stream.
    """
    partial_func = partial(func, *args, **kwargs)
    arg_count = get_partial_argcount(partial_func)
    match arg_count:
        case 1:  # data_item only
            if output_type == "text":

                def matched_func(io: IO, data_item: DataType) -> ReturnType:
                    result = partial_func(data_item)
                    io.write(json.dumps(result) + "\n")
                    return result

            else:
                logger.warning(
                    f"You may need to implement your own output stream handler in processor func if you want to save the output in binary format. "
                    f"Using binary separator: {binary_sep}. Please make sure this is valid for your use case."
                )

                def matched_func(io: IO, data_item: DataType) -> ReturnType:
                    result = partial_func(data_item)
                    assert isinstance(
                        result, bytes
                    ), "The return value of processor func must be bytes if output_type is binary."
                    io.write(result + binary_sep)
                    return result

        case 2:  # io, data_item
            # in this case, the file writing will be handled in the user-defined func
            matched_func = partial_func
        case _:
            raise ValueError(f"func must accept 1 or 2 arguments, got {arg_count}")
    return matched_func


def iterate_wrapper(
    func: Union[
        Callable[Concatenate[DataType, ParamTypes], ReturnType],
        Callable[Concatenate[IO, DataType, ParamTypes], ReturnType],
    ],
    data: Iterable[DataType],
    output: str | IO | None = None,
    output_type: Literal["text", "binary"] = "text",
    restart=False,
    retry=5,
    wait=1,
    on_error: Literal["raise", "continue"] = "raise",
    num_workers=1,
    bar=True,
    flush=True,
    total_items: int | None = None,
    run_name=__name__,
    tmp_dir=default_tmp_dir(),
    binary_sep=b"#iterwrap_sep#",
    binary_load_fn: Callable[[bytes], list[ReturnType]] | None = None,
    envs: list[dict[str, str]] = [],
    *args: ParamTypes.args,
    **kwargs: ParamTypes.kwargs,
) -> Sequence[ReturnType] | None:
    """Wrapper on a processor (func) and iterable (data) to support multiprocessing, retrying and automatic resuming.

    Args:
        func: The processor function. It should accept the following argument patterns: data item only; output stream, data item. Additional args (*args and **kwargs) can be added in func, which should be passed to the wrapper. Within func, the output stream can be used to save data in real time. In text mode: The data should be written to the output stream one sample per line. If the output stream is not provided, the output will be written to a the tmp_dir for progress recovery using jsonl serialization. In binary mode: You can specify a binary separator (default: b"#iterwrap_sep#") or a binary load function (default: None). The return value of func must be bytes and should be written to the output stream in a way compatible with the binary separator or the binary load function. If the output stream is not provided, the output will be written using binary separator.
        data: The data to be processed. It can be an iterable or a sequence. In each iteration, the data item in data will be passed to func.
        output: The output stream. It can be a file path, a file-like object or None. If None, no output will be saved after successful processing.
        output_type: The type of the output stream. Besides the type of the final output, it also affects the temporary file used for progress recovery if output stream is not handled in func. Can be "text" or "binary".
        restart: Whether to restart from the beginning.
        retry: The number of retries for processing each data item.
        on_error: The action to take when an exception is raised in func.
        num_workers: The number of workers to use. If set to 1, the processor will be run in the main process.
        bar: Whether to show a progress bar (package tqdm required).
        flush: Whether to flush the output stream after each data item is processed.
        total_items: The total number of items in data. It is required when data is not a sequence.
        run_name: The name of the run. It is used to construct the checkpoint file path.
        tmp_dir: The dir to save tmp files, including outputs and checkpoints. If None, the tmp files will be saved in the platform tempdir / iterwrap, e.g., /tmp/iterwrap on linux.
        envs: Additional environment variables for each worker. This will be set before spawning new processes.
        *args: Additional positional arguments to be passed to func.
        **kwargs: Additional keyword arguments to be passed to func.

    Returns:
        A list of return values from func.
    """
    # init vars
    if num_workers < 1 or len(envs) and len(envs) != num_workers:
        raise ValueError(
            "num_workers must be a positive integer and envs must be a list of length num_workers"
        )
    if num_workers > 1 and os.name == "nt":
        logger.warning("Iterate wrapper with multiprocessing is untested on Windows; use with caution.")
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

    os.makedirs(tmp_dir, exist_ok=True)
    checkpoints = resume_progress(run_name, restart, num_workers, total_items, tmp_dir)

    wrapped_func = wrap_func(func, output_type, binary_sep, *args, **kwargs)
    retry_func = cast(Callable[[IO, DataType], ReturnType], retry_dec(retry, wait, on_error)(wrapped_func))

    lock = Lock()

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
            tmp_dir,
            checkpoints[i],
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
            returns.extend(p.get_returns(output_type, binary_sep, binary_load_fn))
            p.close()
    else:
        returns = _process_job(*_get_job_args(0))

    # Merge output files if output is specified
    if output is not None:
        if isinstance(output, str):
            dir = os.path.dirname(output)
            if dir and not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            f = open(output, "w" if restart else "a")
        else:
            f = output
        merge_files([get_output_path(run_name, i, tmp_dir) for i in range(num_workers)], f, output_type)
        if isinstance(output, str):
            f.close()

    clean_up(run_name, num_workers, tmp_dir)
    return returns
