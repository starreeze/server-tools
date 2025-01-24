# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from .function import iterate_wrapper
from .generator import IterateWrapper
from .utils import bind_cache_json, check_unfinished, retry_dec

# package info
__version__ = "0.3.1"
__author__ = "Starreeze"
__license__ = "GPLv3"
__url__ = "https://github.com/starreeze/server-tools"
