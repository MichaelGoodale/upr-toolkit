import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="upr-toolkit",
    version="0.0.1",
    author="Michael Goodale",
    author_email="michael.goodale@mail.mcgill.ca",
    description="Unsupervised Phonological Toolkit to analyse phonological representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michaelgoodale/upr-toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
