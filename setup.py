#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='astro_dynamo',
      version='0.1',
      description='Adapting N-body models using pytorch',
      author='Chris Wegg',
      author_email='chriswegg+astrodynamo@gmail.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/cwegg/astro-dynamo',
      packages=find_packages(),
      install_requires=['torch', 'numpy', 'matplotlib', 'scipy', 'pandas'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6'
      )
