from setuptools import find_packages, setup

setup(
    name='poregen',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'einops',
        'tqdm',
        'safetensors',
        'porespy',
        'numpy',
        'matplotlib',
        'diffusers',
        'lightning',
        'transformers',
        'netCDF4',
        'jaxtyping'
    ],
    version='0.1.0',
    description='Neural networks for porous media',
    author='UFRJ',
    license='BSD-3',
)
