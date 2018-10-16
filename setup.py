from setuptools import setup, find_packages

FULL_VERSION = '1.0.0'

with open('README.md') as f:
    readme = f.read()

setup(
    name='cycle_gan',
    version=FULL_VERSION,
    description='Cycle GAN tensorflow implementation.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='aushio@keio.jp',
    packages=find_packages(exclude=('datasets')),
    include_package_data=True,
    test_suite='scripts',
    install_requires=[
        # 'tensorflow-gpu==1.10.1',
        'tensorflow-gpu',
        'numpy',
        'pandas',
        'Pillow',
        'toml',
        'scipy'
    ]
)