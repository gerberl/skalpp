"""with help from https://python-packaging.readthedocs.io/en/latest/minimal.html"""

from setuptools import setup

setup(
    name='skalpp',
    version='0.1',
    description='Some utility functions for `sklearn data` pre-processing. At the moment, mostly, for my own projects. Hopefully, in the future, sufficiently generic and reusable for others to use as well.',
    url='https://github.com/gerberl/skalpp',
    author='Luciano Gerber',
    author_email='L.Gerber@mmu.ac.uk',
    license='BSD-3',
    packages=['skalpp'],
    zip_safe=False
)