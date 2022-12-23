#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Date    : 2022-11-24 19:40:25
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import os, shutil, json
from pathlib import Path
from argparse import ArgumentParser

__version__ = "0.2"
base_path = os.environ["SOFTWARE_BASE"]
status_path = os.path.join(base_path, "var/xpkg-status.json")
tmp_path = os.path.join(base_path, "tmp")
links = {"bin": "usr/bin", "sbin": "usr/sbin", "lib": "usr/lib", "lib64": "usr/lib"}

shell_config = """
export SOFTWARE_BASE={}
export PATH=$SOFTWARE_BASE/bin:$SOFTWARE_BASE/sbin:$PATH
LD_BASE=$SOFTWARE_BASE/usr/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH$( find $LD_BASE -type d -printf ":%p" )
""".format(
    base_path
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--version", "-v", action="version", version="xpkg version " + __version__
    )
    parser.add_argument("--install", "-i", nargs="+", default=[], type=str)
    parser.add_argument("--remove", "-r", nargs="+", default=[], type=str)
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="force install ignoring dependency errors",
    )
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--list", "-l", action="store_true")
    return parser.parse_args()


def fix_link():
    for k, v in links.items():
        src = os.path.join(base_path, k)
        if not Path(src).is_symlink():
            dst = os.path.join(tmp_path, v)
            shutil.copytree(src, dst)
            shutil.rmtree(src)
            os.symlink(dst, src)


def init():
    Path(base_path).mkdir(parents=True, exist_ok=True)
    (Path(base_path) / "tmp").mkdir(exist_ok=True)
    (Path(base_path) / "usr").mkdir(exist_ok=True)
    (Path(base_path) / "usr/bin").mkdir(exist_ok=True)
    (Path(base_path) / "usr/lib").mkdir(exist_ok=True)
    (Path(base_path) / "usr/sbin").mkdir(exist_ok=True)
    (Path(base_path) / "var").mkdir(exist_ok=True)
    (Path(base_path) / "etc").mkdir(exist_ok=True)
    os.symlink(Path(base_path) / "usr/bin", Path(base_path) / "bin", True)
    os.symlink(Path(base_path) / "usr/lib", Path(base_path) / "lib", True)
    os.symlink(Path(base_path) / "usr/lib", Path(base_path) / "lib64", True)
    os.symlink(Path(base_path) / "usr/sbin", Path(base_path) / "sbin", True)

    installed = os.popen("apt list | grep installed").read().splitlines()
    packages = {}
    for name in installed:
        name = name.split("/")[0]
        packages[name] = {"type": "system"}
    with open(status_path, "w") as f:
        json.dump(packages, f, indent=2)

    shutil.copy(__file__, Path(base_path) / "usr/sbin/xpkg")
    os.system("chmod +x {}".format(Path(base_path) / "usr/sbin/xpkg"))
    print("Installation successful. Please put this into your bashrc (or zshrc, etc.):")
    print(shell_config)


def get_depends(name: str) -> list:
    depends = (
        os.popen("apt-cache depends {} | grep ' Depends'".format(name))
        .read()
        .splitlines()
    )
    return [x.split(": ")[1] for x in depends if not x.endswith(">")]


def install_package(name: str, status: dict, force=False, manual=False) -> None:
    if name in status:
        print(name, "is already installed")
        return
    if name.startswith("python3-"):
        if manual:
            print("Error:", name, "is a python3-only package, please use pip instead")
            exit(1)
        else:
            print(
                "Warning: ",
                name,
                "is a python3-only package, using pip instead; note that this won't be recorded by xpkg",
            )
            os.system('python -m pip install "{}"'.format(name))
            return
    depends = get_depends(name)
    if depends:
        print(name, "depends on ", depends, ", installing dependencies first")
    for dep in depends:
        install_package(dep, status, force)
    if os.system("apt-get download " + name):
        if force:
            print("Error installing" + name + ": ignoring")
            return
        else:
            raise RuntimeError("Error installing package " + name)
    filename = [file for file in os.listdir(".") if file.endswith(".deb")][0]
    files = (
        os.popen("dpkg-deb -xv {} {}".format(filename, base_path)).read().splitlines()
    )
    status[name] = {"type": "xpkg", "files": files}
    os.system("rm -f *.deb")


def remove_package(name, status):
    try:
        info = status[name]
    except KeyError:
        print(name, "is not installed")
        exit(1)
    if info["type"] != "xpkg":
        print("Unable to remove system package", name)
        exit(1)
    for file in info["files"]:
        path = os.path.join(base_path, file)
        if os.path.exists(path) and not os.path.isdir(path):
            os.remove(path)
    del status[name]


def main():
    if not os.path.exists(status_path):
        init()
    args = parse_args()
    if args.clear:
        if input("Are you sure to uninstall xpkg and clear all packages? [y/n]") == "y":
            shutil.rmtree(base_path)
            print("Bye!")
        else:
            print("Abort.")
        exit(0)
    if not args.install and not args.remove and not args.list:
        print("type `xpkg --help' for help")
        exit(-1)
    with open(status_path, "r") as f:
        status = json.load(f)
    if args.list:
        print(
            "\n".join(key for key, value in status.items() if value["type"] == "xpkg")
        )
        exit(0)
    for pkg in args.install:
        if len([file for file in os.listdir(".") if file.endswith(".deb")]):
            print(
                "Error: current directory contains deb files. Please cd to another directory"
            )
            exit(-1)
        install_package(pkg, status, args.force, True)
        fix_link()
        print(
            "successfully installed package",
            pkg,
            ", please source .bashrc or restart the shell to use",
        )
    for pkg in args.remove:
        remove_package(pkg, status)
        print("successfully removed package", pkg)
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
