# -*- coding: utf-8 -*-
# @Date    : 2023-12-28 11:37:41
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import sys, os, datetime, json, subprocess, time, logging
from argparse import ArgumentParser
from multiprocessing import Process

log_format = "%(asctime)s --- %(levelname)s: %(message)s"
formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler("jq_log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

output_dir = "/workspace/hal/jq_output"
status_path = "/workspace/hal/jq_status.json"


class JobQueueManager:
    def __init__(self, filename=status_path):
        self.filename = filename
        self.job_id_counter = 0
        self.gpu_free_since: list[None | datetime.datetime] = [None] * 16
        self.gpu_avail: set[int] = set()
        self.gpu_pending: set[int] = set()
        self.gpu_inuse: dict[int, int] = {}
        self.running_jobs: dict[int, Process] = {}

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
        if not self.gpu_pending:
            return
        try:
            gpustat_output = subprocess.check_output(["gpustat", "--json"], text=True)
            gpu_data = json.loads(gpustat_output)
            current_time = datetime.datetime.now()
            for gpu_id, gpu in enumerate(gpu_data["gpus"]):
                if gpu_id not in self.gpu_pending:
                    continue
                if gpu["memory.used"] < 1000 and gpu["utilization.gpu"] == 0:
                    logger.info(f"find free gpu: {gpu_id}")
                    if args.interval == 0:
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
                    elif self.gpu_free_since[gpu_id] is None:
                        self.gpu_free_since[gpu_id] = current_time
                    elif (current_time - self.gpu_free_since[gpu_id]).total_seconds() >= 60 * args.interval:  # type: ignore
                        # logger.info(f"starting job after waited...")
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
                else:
                    self.gpu_free_since[gpu_id] = None
        except subprocess.CalledProcessError:
            logger.error("Error running gpustat.")

    def execute_job(self, job, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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

            logger.info(f"\nJob {job['id']} executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"\nJob {job['id']} failed to execute. Error: {e}")

    def daemon(self, args):
        while True:
            # check complete jobs and release gpus
            for job_id, process in self.running_jobs.copy().items():
                if not process.is_alive():
                    gpu_id = self.gpu_inuse.pop(job_id)
                    self.gpu_avail.add(gpu_id)
                    logger.info(f"job {job_id} finished, gpu {gpu_id} is now available.")
                    process.join()
                    self.running_jobs.pop(job_id)

            # start new jobs
            job_queue = self.load_jobs()
            if len(job_queue):
                self.check_gpu_availability(args)
                if len(self.gpu_avail):
                    job = job_queue[0]
                    gpu_id = self.gpu_avail.pop()
                    logger.info(f"starting job {job['id']} on gpu {gpu_id}")
                    self.gpu_inuse[job["id"]] = gpu_id
                    process = Process(target=self.execute_job, args=(job, gpu_id))
                    self.running_jobs[job["id"]] = process
                    process.start()
                    job_queue = self.load_jobs()
                    job_queue.pop(0)
                    self.save_jobs(job_queue)
                else:
                    logger.info(f"No available gpu.")

            logger.debug(f"gpu available: {self.gpu_avail}, pending: {self.gpu_pending}, inuse: {self.gpu_inuse}")
            time.sleep(args.frequency)

    def start_daemon(self, args):
        os.makedirs(output_dir, exist_ok=True)
        self.gpu_pending = set(args.gpus)
        logger.info("Daemon started.")
        self.daemon(args)


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
    else:
        print("Invalid command or arguments.")


if __name__ == "__main__":
    main()
