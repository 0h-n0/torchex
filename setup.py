import os
import re
import setuptools
from pathlib import Path

p = Path(__file__)

def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)

version = get_version('torchex')

setup_requires = [
    'numpy',
    'pytest-runner'
]

install_requires = [
    'tensorboardx'
]
test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest'
]

setuptools.setup(
    name="torchex",
    version=version,
    python_requires='>3.6',    
    author="Koji Ono",
    author_email="koji.ono@exwzd.com",
    description="Pytorch Extension Module.",
    url='https://github.com/0h-n0/torchex',
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=test_require,
    extras_require={
        'docs': [
            'torch',
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
