import sys
import os
from pathlib import Path


def setup_system_paths():
    setup_cuda_paths()
    setup_classpath()


def setup_cuda_paths():
    dllpath = Path(
        "C:") / "Program Files" / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v11.2" / "bin"
    if dllpath.exists():
        dllstring = str(dllpath.resolve())
        os.add_dll_directory(dllstring)


def setup_classpath():
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
