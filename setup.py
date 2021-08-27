import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="momentchi2",
    version="0.1.8",
    author="Dean Bodenham",
    author_email="deanbodenhampkgs@gmail.com",
    description="A collection of methods for computing the cdf of a weighted sum of chi-squared random variables.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/momentchi2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires = ["scipy>=1", "numpy>=1"]
)

