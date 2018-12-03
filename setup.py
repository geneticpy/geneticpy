from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='geneticpy',
    version='0.0.1.dev',
    packages=find_packages(),
    url='https://github.com/geneticpy/geneticpy',
    license='MIT',
    author='Brandon Schabell',
    author_email='brandonschabell@gmail.com',
    description='An optimization package based on a genetic algorithm.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 0.0.1 - Alpha',
        'License :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Genetic Algorithm :: Hyperparameter Optimization',
      ],
    install_requires=['numpy'],
    tests_require=['pytest', 'pytest-runner']
)
