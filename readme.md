server scripts collection.

# xpkg

xpkg - a minimal package management tool without root

version 0.3 by _xsy_, GPLv3 License

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

This is written by GPT4 using the following prompt:

> Write a python script (jq.py) to maintain a job queue.
>
> It support the following operations:
>
> 1. add: `python jq.py add [job_cmd]`. Add a job to the queue.
> 2. list: `python jq.py ls`. List all jobs in the queue, including an ID (which is a unique integer, starting from 0 and adds 1 for each job), the datetime upon creation, and job command.
> 3. delete: `python jq.py del [job_id]`. Delete a job from the queue by its ID.
> 4. start: `python jq.py start`. Start a daemon to run the jobs in the queue. The daemon should run indefinitely until stopped manually. Note that even after the daemon is started, you can still modify the job queue, such as adding or deleting jobs to the queue.
> 5. stop: `python jq.py stop`. Stop the daemon.
>
> The function of the script is this:
>
> When there is a free GPU available for more than 10 minutes among the first 4 GPUs, it should be sequentially executed by a worker process, with the environment variable `CUDA_AVAILABLE_DEVICES` set to the free GPU ID. The output of the job should be both logged to the console and a file.
