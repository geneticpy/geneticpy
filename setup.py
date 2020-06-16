from setuptools import setup, find_packages


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

test_requirements = [
    'pytest'
]

setup(
    name='geneticpy',
    version='1.2.1',
    packages=find_packages(),
    url='https://github.com/geneticpy/geneticpy',
    download_url='https://github.com/geneticpy/geneticpy/archive/v1.2.1.tar.gz',
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
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Utilities'
    ],
    python_requires='~=3.6',
    install_requires=[
        'numpy>=1.14.0',
        'tqdm',
        'scikit-learn>=0.23.1'
    ],
    tests_require=test_requirements,
    setup_requires=[
        'pytest-runner'
    ],
    extras_require={
        'tests': test_requirements
    },
)
