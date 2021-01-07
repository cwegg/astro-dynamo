#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='astro_dynamo',
      version='0.1',
      description='Tools for making dynamical models',
      author='Chris Wegg',
      author_email='chriswegg+astrodynamo@gmail.com',
      url='https://gitlab.com/chriswegg/astro-dynamo',
      packages=find_packages(),
      install_requires=['torch', 'numpy', 'matplotlib', 'scipy', 'pandas']
      )
