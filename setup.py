from pathlib import Path
import sys

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent

compile_args = ["-O3", "-std=c++14", "-fPIC"]
link_args = []
if sys.platform != "win32":
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

extension = Extension(
    "qdes.utils_cside",
    sources=["csrc/utils_cside_new.cpp"],
    include_dirs=[
        "/usr/include/eigen3",
        "pybind11",
    ],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

try:
    import numpy
    import pybind11
except ImportError as exc:
    raise RuntimeError("Build requires numpy and pybind11") from exc

extension.include_dirs[1] = pybind11.get_include()
extension.include_dirs.append(numpy.get_include())

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[extension],
)
