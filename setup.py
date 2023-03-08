from setuptools import setup, find_packages
 
setup(
   name='crapstack',
   version='0.0.1',
   install_requires=[
      'transformers[torch]==4.25.1',
      'tika',
      'fitz',
      'mmh3',
      'pydantic',
      "faiss-cpu==1.7.2",
      'sqlalchemy',
      'nltk',
      'more-itertools',
      'sentence-transformers',
      'dill',
      'seqeval',
      'jsonschema',
      'quantulum3',
      'langdetect',
      'python-docx',
      "azure-ai-formrecognizer>=3.2.0b2",
      "sqlalchemy>=1.4.2,<2",
      "sqlalchemy_utils",
   #  'farm-haystack[faiss]'
   ],
   license='MIT',
)