import os
from setuptools import setup

requirements_file = os.path.join(os.path.dirname(__file__), 'requirements')
requirements = []
dependency_links = []
with open(requirements_file) as fh:
    for line in fh:
        line = line.strip()
        if line:
            # Make sure the github URLs work here as well
            split = line.split('@')
            split = split[0]
            split = split.split('/')
            url = '/'.join(split[:-1])
            requirement = split[-1]
            requirements.append(requirement)
            # Add the rest of the URL to the dependency links to allow
            # setup.py test to work
            if 'git+https' in url:
                dependency_links.append(line.replace('git+', ''))

setup(
    name='mondac',
    version='0.1.0',
    packages=['mondac.tests', 'mondac', 'mondac.utils'],
    install_requires=requirements,
    dependency_links=dependency_links,
    url='',
    license='GNU Affero General Public License v3',
    author='Jongheon Jeong',
    author_email='jongheonj@exbrain.io',
    description='MONDrian Algorithm Configuration: Module for algorithm configuration with Mondrian forest'
)
