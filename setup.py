from distutils.core import setup

setup(
    name='compton',
    packages=['compton'],
    version='0.1',
    author='Jordan Melendez',
    author_email='jmelendez1992@gmail.com',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ]
)