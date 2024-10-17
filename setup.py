from setuptools import setup, find_packages

# Load the version from __version__.py
version = {}
with open("neuroai_kit/version.py") as fp:
    exec(fp.read(), version)
    
setup(
    name = 'neuroai_kit',
    version=version['__version__'],
    packages = find_packages(),
    install_requires = [
        # "torch",
        # "torchvision"
    ]
)