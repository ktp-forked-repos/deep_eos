from setuptools import setup, find_packages

setup(
    name='deep-eos',
    version='0.0.1',
    description='Deep end-of-sentence detection',
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author='Stefan Schweter',
    author_email='stefan@schweter.it',
    url='https://github.com/stefan-it/deep_eos',
    packages=find_packages(exclude='test'),  # same as name
    license='MIT',
    install_requires=[
        'certifi==2018.10.15',
        'chardet==3.0.4',
        'Click==7.0',
        'idna==2.7',
        'numpy==1.15.4',
        'requests==2.20.1',
        'toml==0.10.0',
        'torch==0.4.1',
        'torchtext==0.3.1',
        'tqdm==4.28.1',
        'urllib3==1.24.1',
    ],
    include_package_data=True,
    python_requires='>=3.6',
)
