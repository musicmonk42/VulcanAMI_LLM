"""Packfile Module - GPK2 format for git-like packfile storage.

Components:
- header: GPK2 header encoding/decoding
- packer: Packfile creation and writing
- reader: Packfile reading and extraction
"""

from .header import GPK2Header, decode_header, encode_header
from .packer import Packer, create_packfile
from .reader import Reader, read_packfile

__all__ = [
    # Header
    "GPK2Header",
    "encode_header",
    "decode_header",
    # Packer
    "Packer",
    "create_packfile",
    # Reader
    "Reader",
    "read_packfile",
]
