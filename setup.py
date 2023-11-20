from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools

    use_setuptools()
    import setuptools

if sys.version_info[0] == 2:
    enum = ["enum34"]
elif sys.version_info[0] == 3:
    enum = []

setuptools.setup(
    name="ai-safety-gridworlds",
    version="2.0",
    description="Extended and multi-objective environments based on DeepMind's "
        "AI Safety Gridworlds. This is a suite of reinforcement learning "
        "environments illustrating various safety properties of intelligent agents.",
    long_description=(
        "Extended and multi-objective environments based on DeepMind's "
        "AI Safety Gridworlds. "
        "This is a suite of reinforcement learning environments illustrating "
        "various safety properties of intelligent agents. These environments "
        "are implemented in pycolab, a highly-customisable gridworld game "
        "engine with some batteries included."
    ),
    url="https://github.com/levitation-opensource/ai-safety-gridworlds/",
    author="Roland Pihlakas, forked from David Lindner, n0p2, and from DeepMind Technologies",
    author_email="roland@simplify.ee",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Console :: Curses",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Testing",
    ],
    keywords=(
        "ai "
        "artificial intelligence "
        "ascii art "
        "game engine "
        "gridworld "
        "gym "
        "reinforcement learning "
        "retro retrogaming "
        "rl "
    ),
    install_requires=[
      "absl-py", 
      "gym",
      "matplotlib",
      "numpy", 
      "pillow",
      "pycolab", 
    ] + enum,
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    test_suite="ai_safety_gridworlds.tests",
    tests_require=["tensorflow"],
    package_data={"ai_safety_gridworlds.helpers": ["*.ttf"]},
)
