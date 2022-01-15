import os

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

test_requirements = [
    'pytest',
    'pandas'
]

setup(
    name='geneticpy',
    packages=find_packages(),
    url='https://github.com/geneticpy/geneticpy',
    license='MIT',
    author='Brandon Schabell',
    author_email='brandonschabell@gmail.com',
    description='GeneticPy is an optimizer that uses a genetic algorithm to quickly search through custom parameter spaces for optimal solutions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Utilities'
    ],
    python_requires='~=3.6',
    install_requires=[
        'numpy>=1.14.0',
        'tqdm',
        'scikit-learn>=0.23.2'
    ],
    tests_require=test_requirements,
    setup_requires=[
        'pytest-runner',
        'setuptools-git-versioning'
    ],
    extras_require={
        'tests': test_requirements
    },
    setuptools_git_versioning={
        'enabled': True,
    }
)
