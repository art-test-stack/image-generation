from enum import Flag, auto

class DataMode(Flag):
    MONO = 0
    COLOR = auto()
    BINARY = auto()
    MISSING = auto()