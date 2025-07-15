# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from .argparser import HfArgumentParser  # noqa: F401
from .function import iterate_wrapper  # noqa: F401
from .generator import IterateWrapper  # noqa: F401
from .utils import bind_cache_json, check_unfinished, retry_dec  # noqa: F401

# package info
__version__ = "0.4.3"
__author__ = "Starreeze"
__license__ = "GPLv3"
__url__ = "https://github.com/starreeze/server-tools"
