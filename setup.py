from setuptools import setup, find_packages

setup(
    name='ml_framework',
    version='0.1.0',
    description='A simple, reusable data science toolkit for ML experiments',
    author='Ben Harvey',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
