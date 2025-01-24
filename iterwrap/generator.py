import os
from itertools import product
from typing import Generic, Iterable, Literal, Sequence

from tqdm import tqdm

from .utils import DataType, default_tmp_dir, get_checkpoint_path


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
        tmp_dir=default_tmp_dir(),
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
            tmp_dir: temporary directory for checkpoint files
        """
        if mode == "product":
            self.data: Sequence[DataType] = convert_type(product(*data))
        elif mode == "zip":
            self.data = convert_type(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")
        total_items = total_items if total_items is not None else len(self.data)
        os.makedirs(tmp_dir, exist_ok=True)
        checkpoint_path = get_checkpoint_path(run_name, tmp_dir)
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
