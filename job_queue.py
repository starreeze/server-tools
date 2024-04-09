# -*- coding: utf-8 -*-
# @Date    : 2023-12-28 11:37:41
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import datetime
import json
import os
import subprocess
import time
import logging
import sys
from argparse import ArgumentParser, Namespace

log_format = "%(asctime)s --- %(levelname)s: %(message)s"
formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# gpu_list = [0, 1, 2, 3]
# free_minutes_before_run = 3
output_dir = "/home/xingsy/jq_output"
status_path = "/home/xingsy/jq_status.json"


class JobQueueManager:
    def __init__(self, filename=status_path):
        self.filename = filename
        self.job_id_counter = 0
        self.gpu_free_since: list[None | datetime.datetime] = [None] * 16

    def load_jobs(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                return json.load(file)
        return []

    def save_jobs(self, job_queue):
        with open(self.filename, "w") as file:
            json.dump(job_queue, file)

    def add_job(self, job_cmd: list[str]):
        job_queue = self.load_jobs()
        job_id = max([job["id"] for job in job_queue], default=-1) + 1
        current_dir = os.getcwd()
        job = {
            "id": job_id,
            "datetime": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "command": job_cmd,
            "dir": current_dir,
        }
        job_queue.append(job)
        self.save_jobs(job_queue)
        print(f"Added job {job_id}.")

    def list_jobs(self):
        job_queue = self.load_jobs()
        for job in job_queue:
            print(f"{job['id']: 03d} ({job['datetime']}): {job['command']}")

    def delete_job(self, job_id):
        job_queue = self.load_jobs()
        job_queue = [job for job in job_queue if job["id"] != job_id]
        self.save_jobs(job_queue)
        print(f"Deleted job {job_id}.")

    def check_gpu_availability(self, args):
        try:
            gpustat_output = subprocess.check_output(["gpustat", "--json"], text=True)
            gpu_data = json.loads(gpustat_output)
            current_time = datetime.datetime.now()
            for gpu_id, gpu in enumerate(gpu_data["gpus"]):
                if gpu_id not in args.gpus:
                    continue
                if gpu["memory.used"] < 1000:
                    logger.info(f"find free gpu: {gpu_id}")
                    if args.interval == 0:
                        return gpu_id
                    if self.gpu_free_since[gpu_id] is None:
                        self.gpu_free_since[gpu_id] = current_time
                    elif (current_time - self.gpu_free_since[gpu_id]).total_seconds() >= 60 * args.interval:  # type: ignore
                        logger.info(f"starting job after waited...")
                        return gpu_id
                else:
                    self.gpu_free_since[gpu_id] = None
            logger.info(f"Not starting job.")
            return None
        except subprocess.CalledProcessError:
            logger.error("Error running gpustat.")
            return None

    def execute_job(self, job, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        output_file = os.path.join(output_dir, f"job_{job['id']}_output.txt")
        original_dir = os.getcwd()
        os.chdir(job["dir"])
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
                proc.wait()

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, job["command"])

            logger.info(f"\nJob {job['id']} executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"\nJob {job['id']} failed to execute. Error: {e}")
        finally:
            os.chdir(original_dir)

    def daemon(self, args):
        while True:
            job_queue = self.load_jobs()
            if len(job_queue):
                gpu_id = self.check_gpu_availability(args)
                if gpu_id is not None:
                    job = job_queue[0]
                    self.execute_job(job, gpu_id)
                    job_queue = self.load_jobs()
                    job_queue.pop(0)
                    self.save_jobs(job_queue)
            time.sleep(args.frequency)

    def start_daemon(self, args):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Daemon started.")
        self.daemon(args)

    def stop_daemon(self):
        pass  # XXX


def parse_start_args(sys_args: list[str]):
    parser = ArgumentParser()
    parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--interval", "-i", type=int, default=0, help="wait time before using a gpu")
    parser.add_argument("--frequency", "-f", type=int, default=30, help="frequency to check for gpu")
    return parser.parse_args(sys_args)


def main():
    manager = JobQueueManager()
    op, args = sys.argv[1], sys.argv[2:]
    if op == "add" and args:
        manager.add_job(args)
    elif op == "ls":
        manager.list_jobs()
    elif op == "del" and args:
        for id in args:
            manager.delete_job(int(id))
    elif op == "start":
        args = parse_start_args(args)
        manager.start_daemon(args)
    elif op == "stop":
        manager.stop_daemon()
    else:
        print("Invalid command or arguments.")


if __name__ == "__main__":
    main()
