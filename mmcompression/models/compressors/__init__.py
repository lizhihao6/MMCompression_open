# Copyright (c) NJU Vision Lab. All rights reserved.
from .jpeg import JPEGCompressor
from .mpeg import MPEGCompressor
from .nic import NICCompressor

__all__ = ['NICCompressor', 'MPEGCompressor', 'JPEGCompressor']
