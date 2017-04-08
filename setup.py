# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='EMN',
    version='0.0.1',
    description='Edge sharing Motif Networks Robusts',
    long_description=readme,
    author='Moinuddin K Syed',
    author_email='lnusk@vcu.com',
    url='https://github.com/syedkm/EdgeConnectedMotifs',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
