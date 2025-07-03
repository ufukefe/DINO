from setuptools import setup, find_packages
setup(
    name="dino-models",
    version="0.1.0",
    description="DINO object-detector research code",
    packages=find_packages(),
    install_requires=["torch"],   # add others if the repo needs them
)
