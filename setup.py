from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [x.strip() for x in content
                if x.strip()
                and not x.startswith('#')
                and not x.startswith('http')]

dependency_links = [x.strip() for x in content
                    if x.strip() and x.startswith('http')]

setup(name='fintell',
      version="0.0.1",
      description="fintell Model",
      license="MIT",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      #url="https://github.com/victor-betus/fintell",
      install_requires=requirements,
      dependency_links=dependency_links,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
