import json
import logging
import os
import traceback
from functools import wraps
from glob import glob
from multiprocessing import synchronize
from time import sleep
from typing import IO, Callable, Literal, ParamSpec, TypeVar

from tqdm import tqdm

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


logger = setup_tqdm_logger()


def check_unfinished(run_name: str):
    "To check if the run is unfinished, according to whether the checkpoint file and the cache file exists"
    ckpt = ckpt_tmpl.format(name=run_name)
    if os.path.exists(ckpt):
        num_cache = len(glob(output_tmpl.format(name=run_name, id="*")))
        num_ckpt = len(open(ckpt, "r").readlines())
        if num_cache == num_ckpt:
            return True
        logger.warning(f"unmatched: {num_cache} unfinished files vs {num_ckpt} checkpoints, restart from the beginning")
    return False


def retry_dec(retry=5, wait=1, on_error: Literal["raise", "continue"] = "raise"):
    "decorator for retrying a function on exception; on_error could be raise or continue"

    def decorator(func: Callable[ParamTypes, ReturnType]):
        def wrapper(*args: ParamTypes.args, **kwargs: ParamTypes.kwargs) -> ReturnType | None:
            if retry <= 1:
                return func(*args, **kwargs)
            for j in range(retry):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt as e:
                    logger.error(traceback.format_exc())
                    raise e
                except BaseException as e:
                    if j == retry - 1:
                        if on_error == "raise":
                            logger.error("All retry failed:")
                            logger.info(traceback.format_exc())
                            raise e
                        elif on_error == "continue":
                            logger.warning(
                                f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy."
                            )
                            return
                    logger.warning(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")
                    logger.debug(traceback.format_exc())
                    sleep(wait)

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


def write_ckpt(path: str, checkpoint: int, process_idx: int, lock: synchronize.Lock):
    with lock:
        with open(path, "r") as f:
            checkpoints = f.read().splitlines()
        checkpoints[process_idx] = str(checkpoint)
        with open(path, "w") as f:
            f.write("\n".join(checkpoints))


def merge_files(input_paths: list[str], output_stream: IO | None):
    for path in input_paths:
        if output_stream is not None:
            with open(path, "r") as f:
                output_stream.write(f.read())
        if os.path.exists(path):
            os.remove(path)


def clean_up(run_name: str, num_workers: int):
    "Clean up the checkpoint and temporary result files"
    checkpoint_path = ckpt_tmpl.format(name=run_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    for i in range(num_workers):
        result_path = output_tmpl.format(name=run_name, id=i)
        if os.path.exists(result_path):
            os.remove(result_path)


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
                logger.info(f"Loading cached file {file_path} from disk...")
                with open(file_path, "r") as f:
                    return json.load(f)

        return wrapper

    return decorator
