#################### Maintained by Hatch ####################
# This file is auto-generated by hatch. If you'd like to customize this file
# please add your changes near the bottom marked for 'USER OVERRIDES'.
# EVERYTHING ELSE WILL BE OVERWRITTEN by hatch.
#############################################################
from io import open

from setuptools import setup, find_packages
import os

with open('mimikit/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), "r", encoding="utf-8") as f:
    REQUIRES = [ln.strip() for ln in f.readlines() if ln.strip()]

PACKAGES = find_packages(exclude=('tests', 'tests.*'))

if os.environ.get("MIMIKIT_NOTORCH", False):
    PACKAGES = [p for p in PACKAGES if "data" in p or "connectors" in p]
    REQUIRES = [r for r in REQUIRES if "torch" not in r and "test-tube" not in r]

kwargs = {
    'name': 'mimikit',
    'version': version,
    'description': 'Python module for generating audio with neural networks',
    'long_description': readme,
    "long_description_content_type": "text/markdown",
    'author': 'Antoine Daurat',
    'author_email': 'ktonalberlin@gmail.com',
    'url': 'https://github.com/k-tonal/mimikit',
    'download_url': 'https://github.com/k-tonal/mimikit',
    # 'license': 'GNU General Public License v3 (GPLv3)',
    'classifiers': [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "audio music sound deep-learning",
    'python_requires': '>=3.6',
    'install_requires': REQUIRES,
    'tests_require': ['coverage', 'pytest'],
    'packages': PACKAGES,
    "entry_points": {
        'console_scripts': [
            'freqnet-db=mimikit.data.freqnet_db:main'
        ]}

}

setup(**kwargs)
