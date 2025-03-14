import json
import logging
import os
import tempfile
import traceback
from functools import wraps
from glob import glob
from inspect import signature
from time import sleep
from typing import IO, Callable, Literal, ParamSpec, TypeVar

from multiprocess import synchronize
from tqdm import tqdm


# default file path
def default_tmp_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "iterwrap")


def get_output_path(name: str, id: int | Literal["*"], tmp_dir: str) -> str:
    return os.path.join(tmp_dir, f"{name}_p{id}.output")


def get_checkpoint_path(name: str, tmp_dir: str) -> str:
    return os.path.join(tmp_dir, f"{name}.ckpt")


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


def check_unfinished(run_name: str, tmp_dir: str | None = None):
    "To check if the run is unfinished, according to whether the checkpoint file and the cache file exists"
    if tmp_dir is None:
        tmp_dir = default_tmp_dir()
    ckpt = get_checkpoint_path(run_name, tmp_dir)
    if os.path.exists(ckpt):
        num_cache = len(glob(get_output_path(run_name, "*", tmp_dir)))
        num_ckpt = len(open(ckpt, "r").readlines())
        if num_cache == num_ckpt:
            return True
        logger.warning(
            f"unmatched: {num_cache} unfinished files vs {num_ckpt} checkpoints, restart from the beginning"
        )
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


def merge_files(input_paths: list[str], output_stream: IO | None, output_type: Literal["text", "binary"]):
    open_flag = "r" if output_type == "text" else "rb"
    for path in input_paths:
        if output_stream is not None:
            with open(path, open_flag) as f:
                output_stream.write(f.read())
        if os.path.exists(path):
            os.remove(path)


def clean_up(run_name: str, num_workers: int, tmp_dir: str):
    "Clean up the checkpoint and temporary result files"
    checkpoint_path = get_checkpoint_path(run_name, tmp_dir)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    for i in range(num_workers):
        result_path = get_output_path(run_name, i, tmp_dir)
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


def get_partial_argcount(f: Callable) -> int:
    """
    Get the number of arguments that haven't been provided yet for a variable with __call__ method.
    Args:
        f: the object to be checked. Can be a partial function, a vanilla function,or any other object with __call__ method.
    Returns:
        the number of arguments that haven't been provided yet
    """
    if not hasattr(f, "func"):
        # If it's not a partial function, return the number of parameters
        # For classes with __call__, we need to check the __call__ method
        if (
            hasattr(f, "__call__")
            and not isinstance(f, type)
            and type(f).__call__ is not object.__call__
            and not callable(getattr(f, "__call__", None))
        ):
            sig = signature(f.__call__)
            # Check if __call__ is a regular method (has self), classmethod (has cls), or staticmethod (has neither)
            if isinstance(type(f).__call__, (classmethod, staticmethod)):
                self_param = 0  # No need to subtract for classmethods or staticmethods
            else:
                self_param = 1  # Subtract 1 for 'self'
            return len(sig.parameters) - self_param
        return len(signature(f).parameters)

    # For partial functions, we need to check how many arguments are still required
    # Get the original function (may be nested partials)
    orig_func = f
    while hasattr(orig_func, "func"):
        orig_func = orig_func.func

    # Get the signature of the original function
    # For classes with __call__, we need to check the __call__ method
    if (
        hasattr(orig_func, "__call__")
        and not isinstance(orig_func, type)
        and type(orig_func).__call__ is not object.__call__
        and not callable(getattr(orig_func, "__call__", None))
    ):
        sig = signature(orig_func.__call__)
        # Check if __call__ is a regular method (has self), classmethod (has cls), or staticmethod (has neither)
        if isinstance(type(orig_func).__call__, (classmethod, staticmethod)):
            self_param = 0  # No need to subtract for classmethods or staticmethods
        else:
            self_param = 1  # Subtract 1 for 'self'
    else:
        sig = signature(orig_func)
        self_param = 0

    # Count how many arguments have been provided through partial application
    provided_args = set()
    current = f
    while hasattr(current, "func"):
        # Add positional args
        # Note: due to the flattening of partial implementation in CPython, update on range is ok
        provided_args.update(range(len(current.args)))
        # Add keyword args
        provided_args.update(current.keywords.keys())
        current = current.func

    # Return the number of parameters that haven't been provided yet
    return len(sig.parameters) - len(provided_args) - self_param
