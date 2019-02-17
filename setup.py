
import os
import sys
from setuptools import setup, Command


_python = 'python%d' % sys.version_info.major


class _TestCommand(Command):
    user_options = [
        ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        run_str = "%s -m unittest discover test *test.py" % _python
        os.system(run_str)


setup(
    name='adversarial_autoencoder_classifier',
    version='0.0.1',
    author='Shahar Azulay, Rinat Ishak',
    author_email='shahar4@gmail.com',
    url='https://github.com/shaharazulay/adversarial-autoencoder-classifier',
    packages=[
        'source'
    ],
    license='bsd',
    description='Adversarial Autoencoder Classifier',
    long_description=open('docs/README.rst').read(),
    install_requires=[],
    zip_safe=False,
    package_data={},
    include_package_data=True,
    cmdclass={
        'test': _TestCommand,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
