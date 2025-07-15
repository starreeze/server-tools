import os
from itertools import product
from typing import Iterable, Literal, Sequence

from tqdm import tqdm

from .utils import default_tmp_dir, get_checkpoint_path


class IterateWrapper:
    def __init__(
        self,
        *data: Iterable,
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
        if len(data) == 1:
            self.data: Sequence = convert_type(data[0])
        elif mode == "product":
            self.data: Sequence = convert_type(product(*data))
        elif mode == "zip":
            self.data: Sequence = convert_type(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")

        os.makedirs(tmp_dir, exist_ok=True)
        self.checkpoint_path = get_checkpoint_path(run_name, tmp_dir)
        self.restart = restart
        self.bar = bar
        self.total_items = total_items

    def __iter__(self):
        if self.restart:
            os.remove(self.checkpoint_path)
        try:
            with open(self.checkpoint_path, "r") as checkpoint_file:
                checkpoint = int(checkpoint_file.read().strip())
        except FileNotFoundError:
            checkpoint = 0

        total_items = self.total_items if self.total_items is not None else len(self.data)
        if self.bar >= 0:
            wrapped_range = tqdm(
                range(checkpoint, total_items), initial=checkpoint, total=total_items, position=self.bar
            )
        else:
            wrapped_range = range(checkpoint, total_items)
        self.wrapped_iter = iter(wrapped_range)
        self.index = 0
        return self

    def __next__(self):
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(self.checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        with open(self.checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(self.index))
        return self.data[self.index]
