#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-12-28 11:37:41
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
A simple gpu job queue. This will maintain a queue of jobs to be executed on gpus, and will run them sequentially,
always beginning from the job with highest orders. Among the same order,
those jobs with higher gpu demands will be prioritized. If there is no free GPU available,
it wait until there are enough free GPUs to execute the job.
It will automatically set the environment variable `CUDA_AVAILABLE_DEVICES` to the free GPU IDs when run the job.
The output of the jobs will be both logged to the console and a file.

It support the following operations:
1. add: `python jobq.py add [job_cmd] [-n n_gpus_required] [-o job_order]`.
    Add a job to the queue with command to be executed with os.system(job_cmd).
2. list: `python jobq.py ls`. List all jobs in the queue, including an ID
    (which is a unique integer, starting from 0 and adds 1 for each job),
    the datetime upon creation, job command, and the gpu requirement.
3. delete: `python jobq.py del [job_id]`. Delete a job from the queue by its ID.
4. start: `python jobq.py start [-g gpus_to_be_used] [-i wait_time_before_using_a_free_gpu]
    [-f pool_frequency_for_checking_gpu_availability]`.
    Start a daemon to run the jobs in the queue. The daemon should run indefinitely until stopped manually.
    Note that even after the daemon is started, you can still modify the job queue,
    such as adding or deleting jobs to the queue.
"""

from __future__ import annotations
import sys, os, datetime, json, subprocess, time, logging
from typing import Iterable, Any
from argparse import ArgumentParser
from multiprocessing import Process

output_dir = os.path.expanduser("/home/nfs04/xingsy/logs/jq_output")
status_path = os.path.expanduser("/home/nfs04/xingsy/logs/jq_status.json")
logger = logging.getLogger()


def setup_logger():
    log_format = "%(asctime)s --- %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(output_dir, "jq.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class GPUManager:
    def __init__(self, gpus: Iterable[int]) -> None:
        self.gpu_avail: set[int] = set()
        self.gpu_pending: set[int] = set(gpus)
        self.gpu_inuse: dict[int, set[int]] = {}  # job_id -> {gpu_id}
        self.gpu_free_since: dict[int, datetime.datetime] = {}  # gpu_id -> timestamp

    def update_gpu_status(self, interval: int) -> None:
        if not self.gpu_pending:
            logger.debug(f"gpu available: {self.gpu_avail}, pending: {self.gpu_pending}, inuse: {self.gpu_inuse}")
            return
        try:
            gpustat_output = subprocess.check_output(["gpustat", "--json"], text=True)
        except subprocess.CalledProcessError:
            logger.error("Error running gpustat.")
            return
        gpu_data = json.loads(gpustat_output)
        current_time = datetime.datetime.now()
        for gpu_id, gpu in enumerate(gpu_data["gpus"]):
            if gpu["memory.used"] < 1000 and gpu["utilization.gpu"] == 0:
                if gpu_id in self.gpu_pending:
                    logger.info(f"find free gpu: {gpu_id}")
                    if interval == 0:
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
                    elif gpu_id not in self.gpu_free_since:
                        self.gpu_free_since[gpu_id] = current_time
                    elif (current_time - self.gpu_free_since[gpu_id]).total_seconds() >= 60 * interval:  # type: ignore
                        # logger.info(f"starting job after waited...")
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
            else:
                self.gpu_free_since.pop(gpu_id, None)
                if gpu_id in self.gpu_avail:
                    self.gpu_avail.remove(gpu_id)
                    self.gpu_pending.add(gpu_id)
        logger.debug(f"gpu available: {self.gpu_avail}, pending: {self.gpu_pending}, inuse: {self.gpu_inuse}")

    def allocate(self, job_id: int, ngpu: int) -> list[int] | None:
        "Allocate gpus according to number of gpus needed. Return None if no available."
        if len(self.gpu_avail) < ngpu:
            return None
        allocated = [self.gpu_avail.pop() for _ in range(ngpu)]
        self.gpu_inuse[job_id] = set(allocated)
        return allocated

    def free(self, job_id: int) -> set[int]:
        freed = self.gpu_inuse.pop(job_id)
        self.gpu_avail |= freed
        return freed


class JobQueue:
    def __init__(self, filename=status_path):
        self.filename = filename
        self.job_id_counter = 0
        self.running_jobs: dict[int, Process] = {}
        self.gpu_manager: GPUManager

    def load(self) -> list[dict[str, Any]]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                return json.load(file)
        return []

    def save(self, job_queue: list[dict[str, Any]]):
        job_queue = sorted(job_queue, key=lambda x: (x["order"], -x["ngpu"]))
        with open(self.filename, "w") as file:
            json.dump(job_queue, file)

    def add(self, args):
        job_queue = self.load()
        job_id = max([job["id"] for job in job_queue], default=-1) + 1 if args.id == -1 else args.id
        current_dir = os.getcwd()
        job = {
            "id": job_id,
            "datetime": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "command": args.cmd,
            "ngpu": args.ngpu,
            "order": args.order,
            "dir": current_dir,
        }
        job_queue.append(job)
        self.save(job_queue)
        print(f"Added job {job_id}.")

    def list(self):
        job_queue = self.load()
        for job in job_queue:
            print(
                f"{job['id']: 03d} ({job['datetime']}): {job['command']}, "
                f"gpus required: {job['ngpu']}, job order: {job['order']}"
            )

    def delete(self, job_id):
        job_queue = self.load()
        job_queue = [job for job in job_queue if job["id"] != job_id]
        self.save(job_queue)
        print(f"Deleted job {job_id}.")

    def execute(self, job: dict[str, Any], gpus: list[int]):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        output_file = os.path.join(output_dir, f"job_{job['id']}_output.txt")
        os.chdir(job["dir"])  # chdir in subprocess will not change the dir of main process
        try:
            with open(output_file, "w") as file, subprocess.Popen(
                job["command"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                text=True,
            ) as proc:
                while True:
                    char = proc.stdout.read(1)  # type: ignore
                    if not char:
                        break
                    print(char, end="", flush=True)
                    file.write(char)
                if proc.wait() != 0:
                    raise subprocess.CalledProcessError(proc.returncode, job["command"])

            logger.info(f"Job {job['id']} executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Job {job['id']} failed to execute. Error: {e}")

    def daemon(self, args):
        while True:
            # check complete jobs and release gpus
            for job_id, process in self.running_jobs.copy().items():
                if not process.is_alive():
                    freed = self.gpu_manager.free(job_id)
                    logger.info(f"job {job_id} finished, gpu {freed} is now available.")
                    process.join()
                    self.running_jobs.pop(job_id)

            job_queue = self.load()
            if len(job_queue) == 0:
                time.sleep(args.frequency)
                continue

            # start new jobs
            self.gpu_manager.update_gpu_status(args.interval)
            started_job_idx = []
            for i, job in enumerate(job_queue):
                gpus = self.gpu_manager.allocate(job["id"], job["ngpu"])
                if gpus is None:
                    continue
                logger.info(f"starting job {job['id']} on gpu {gpus}")
                process = Process(target=self.execute, args=(job, gpus))
                self.running_jobs[job["id"]] = process
                process.start()
                started_job_idx.append(i)
            if started_job_idx:
                started_job_idx.reverse()
                for i in started_job_idx:
                    del job_queue[i]
                self.save(job_queue)
            else:
                logger.info(f"No available gpu.")

            time.sleep(args.frequency)

    def start(self, args):
        os.makedirs(output_dir, exist_ok=True)
        setup_logger()
        self.gpu_manager = GPUManager(args.gpus)
        logger.info("Daemon started.")
        self.daemon(args)


class ArgParser:
    def __init__(self, args: list[str] | None):
        self.args = args

    def parse_start(self):
        parser = ArgumentParser()
        parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0, 1, 2, 3])
        parser.add_argument("--interval", "-i", type=int, default=0, help="wait time before using a gpu")
        parser.add_argument("--frequency", "-f", type=int, default=30, help="frequency to check for gpu")
        return parser.parse_args(self.args)

    def parse_add(self):
        parser = ArgumentParser()
        parser.add_argument("cmd", nargs="+", help="the full command for this job")
        parser.add_argument("--ngpu", "-n", type=int, default=1, help="number of gpus needed for this job")
        parser.add_argument("--order", "-o", type=int, default=0, help="the order of this job")
        parser.add_argument("--id", "-i", type=int, default=-1, help="the id of this job")
        known_args, unknown_args = parser.parse_known_args(self.args)
        known_args.cmd.extend(unknown_args)
        return known_args


def print_help(invalid: bool):
    if invalid:
        print("Invalid arguments.")
    print("Usage:")
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_help(invalid=True)
        return
    queue = JobQueue()
    op, args = sys.argv[1], sys.argv[2:]
    parser = ArgParser(args)
    if op == "add" and args:
        queue.add(parser.parse_add())
    elif op == "ls":
        queue.list()
    elif op == "del" and args:
        for id in args:
            queue.delete(int(id))
    elif op == "start":
        queue.start(parser.parse_start())
    else:
        print_help(invalid=not ("-h" in sys.argv[1] or "help" in sys.argv[1]))


if __name__ == "__main__":
    main()
