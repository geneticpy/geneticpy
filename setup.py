from setuptools import setup, find_packages


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='geneticpy',
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/geneticpy/geneticpy',
    download_url='https://github.com/geneticpy/geneticpy/archive/v0.0.2.tar.gz',
    license='MIT',
    author='Brandon Schabell',
    author_email='brandonschabell@gmail.com',
    description='Hyperparameter optimization based on a genetic algorithm.',
    long_description=this_directory,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities'
    ],
    python_requires='~=3.4',
    install_requires=['numpy'],
    tests_require=['pytest', 'pytest-runner']
)
