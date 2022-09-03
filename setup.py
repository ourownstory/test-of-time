import os
import setuptools

dir_repo = os.path.abspath(os.path.dirname(__file__))
# read the contents of REQUIREMENTS file
with open(os.path.join(dir_repo, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()
# read the contents of README file
with open(os.path.join(dir_repo, "README.md"), encoding="utf-8") as f:
    readme = f.read()
# read the version name
with open("tot/_version.py") as f:
    exec(f.read())

setuptools.setup(
    name="test-of-time",
    version=__version__,
    description="Evaluate forecasting models",
    author="Oskar Triebe",
    author_email="trieb@stanford.edu",
    url="https://github.com/ourownstory/test-of-time",
    license="MIT",
    packages=["tot"],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": ["black", "twine", "wheel", "sphinx>=4.2.0", "pytest>=6.2.3", "pytest-cov", "prophet"],
        "full": ["prophet"],
    },
    # setup_requires=[""],
    scripts=["scripts/tot_dev_setup.py"],
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)