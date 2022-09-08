from setuptools import setup, find_packages

from tabex import __version__

setup(
    name='tabex',
    version=__version__,

    url='https://github.com/iliya-malecki/tabex',
    author='Iliya Malecki',
    author_email='iliyamalecki@gmail.com',
    
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'matplotlib',
        'layoutparser',
        'pytesseract >= 0.3.10',
        'opencv-python >= 4.4',
        'docx >= 0.8.11',
        'pandas >= 1.0',
        'numpy >= 1.20',
        'scikit-learn >= 1.0',
    ],
)