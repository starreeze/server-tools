from iterwrap import IterateWrapper, iterate_wrapper


def _perform_operation(item: int):
    from time import sleep

    global _tmp
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
        num_workers=num_workers,
    )
    print(returns)


def _test_wrapper():
    from time import sleep

    for i in IterateWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    _tmp = True
    _test_fn()
