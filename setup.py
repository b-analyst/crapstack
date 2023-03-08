from setuptools import setup, find_packages
 
setup(
   name='crapstack',
   version='0.0.1',
   install_requires=[
    'transformers[torch]==4.25.1',
    'tika',
    'fitz',
   #  'farm-haystack[faiss]'
   ],
   license='MIT',
)