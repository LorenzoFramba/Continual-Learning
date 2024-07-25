from distutils.core import setup
setup(
    name='nsml Seg',
    version='1.0',
    description='nsml',
    install_requires=[
        'torch==2.2.0',
        'visdom',
        'numpy'
    ]
)
