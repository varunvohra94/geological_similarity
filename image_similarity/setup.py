from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'pandas==1.3.0',
    'gcsfs==2021.06.1',
    'numpy==1.21.0',
    'torch==1.9.0',
    'matplotlib==3.4.2',
    'torchvision==0.10.0',
    'tensorboard==2.5.0',
    'google-cloud-storage==1.40.0'
]

setup(
    name='trainer',
    version='0.1',
    author = 'Varun Vohra',
    author_email = 'varun.vohra94@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Image Similarity Training'
)
