import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pywoc',
    version="0.4.2",
    author='Cristiano Sabiu',
    author_email='csabiu@gmail.com',
    description='2-D comparison',
    url='https://github.com/csabiu/woc',
    license='GNU GPL v3 License',
    long_description=read('README'),
    long_description_content_type='text/plain',
    packages=['pywoc'],
    # Keep runtime dependencies minimal: the core algorithm only needs
    # numpy and numba. Plotting and example generation are optional.
    install_requires=['numpy', 'numba'],
    python_requires=">=3.8",
    extras_require={
        'plot': ['matplotlib>=3.3'],
        'examples': ['astropy>=5', 'scipy>=1.7', 'matplotlib>=3.3'],
    },
    # Disable automatic License-File metadata emission for compatibility with
    # older PyPI/Twine validators.
    license_files=[],
    # LICENSE is included via MANIFEST.in; omit license_files to avoid
    # emitting a metadata field some PyPI validators reject for sdists.
    project_urls={
        'Source': 'https://github.com/csabiu/WOC',
        'Bug Tracker': 'https://github.com/csabiu/WOC/issues',
        'Documentation': 'https://github.com/csabiu/WOC#readme',
    },
    keywords=[
        'woc',
        'spatial statistics',
        'statistics',
        'python'
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
        ]
    )
