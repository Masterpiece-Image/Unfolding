
from distutils.core import setup


with open('requirements.txt') as f:
    requirements = f.readlines()


setup(
    name = 'Unfolding',
    version = 'v0.1.0',
    author = 'Jessy Khafif',
    author_email= 'khafifjessy.github@gmail.com',
    packages= [
        'Unfolding'
    ],
    install_requires=requirements
)