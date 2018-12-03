from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='geneticpy',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/geneticpy/geneticpy',
    download_url='https://github.com/geneticpy/geneticpy/archive/v0.0.1.tar.gz',
    license='MIT',
    author='Brandon Schabell',
    author_email='brandonschabell@gmail.com',
    description='Hyperparameter optimization based on a genetic algorithm.',
    long_description=readme(),
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
