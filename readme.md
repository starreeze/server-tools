server scripts collection.

# xpkg

xpkg - a minimal package management tool without root

version 0.3 by _xsy_.

## requirements

ubuntu >= 16.04

python >= 3.5

## features

- No root needed;

- automatically solve dependencies;

- simple and convenient.

## usage

### install

Before the first use, run

```
SOFTWARE_BASE=<path> python xpkg.py
```

in which `<path>` is an arbitrary writable location where you want to put all the packages to be installed.

After complete, you'll see a notice asking you to put something into you .bashrc file (or whatever shell config file). Just do it manually.

Or if already installed an older version, you can run

```
SOFTWARE_BASE=<path> python xpkg.py --update
```

after obtaining the latest version.

### command

After installation, you can use xpkg at your will.

```
xpkg --help           # show help
xpkg --version        # show version
xpkg --list           # list all installed packages
xpkg -i unar aria2    # install packages
xpkg -i unar --force  # force install, i.e. ignoring dependency problems
xpkg -r unar aria2    # remove packages
xpkg --clear          # uninstall xpkg and clear all packages
```

Enjoy!

## Known issues

This script depends on the apt source lists to get packages. However, sometimes these lists are outdated and therefore 404 errors would occur.

The best solution is to ask your administrator to run `apt update`. But if you can, why not just ask him to install the packages for you?

In this case you can try with `--force`, and if that fails we really have nothing to do about it.

# job queue

A simple gpu job queue. This will maintain a queue of jobs to be executed on gpus, and will run them sequentially, always beginning from the job with highest orders. Among the same order, those jobs with higher gpu demands will be prioritized. If there is no free GPU available, it wait until there are enough free GPUs to execute the job. It will automatically set the environment variable `CUDA_AVAILABLE_DEVICES` to the free GPU IDs when run the job. The output of the jobs will be both logged to the console and a file.

It support the following operations:

1. add: `python jobq.py add [job_cmd] [-n n_gpus_required] [-o job_order]`. Add a job to the queue with command to be executed with os.system(job_cmd).
2. list: `python jobq.py ls`. List all jobs in the queue, including an ID (which is a unique integer, starting from 0 and adds 1 for each job), the datetime upon creation, job command, and the gpu requirement.
3. delete: `python jobq.py del [job_id]`. Delete a job from the queue by its ID.
4. start: `python jobq.py start [-g gpus_to_be_used] [-i wait_time_before_using_a_free_gpu] [-f pool_frequency_for_checking_gpu_availability]`. Start a daemon to run the jobs in the queue. The daemon should run indefinitely until stopped manually. Note that even after the daemon is started, you can still modify the job queue, such as adding or deleting jobs to the queue.

# iterator wrapper

Wrapper on an iterable to support interruption & auto resume, retrying and multiprocessing.

There are three APIs provided:

1. IterateWrapper: wrap some iterables to provide automatic resuming on interruption, no retrying and limited to sequence
2. retry_dec: decorator for retrying a function on exception
3. iterate_wrapper: need hand-crafted function but support retrying, multiprocessing and iterable.

See the source code for usage.
