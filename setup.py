import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="momentchi2",
    version="0.1.0",
    author="Dean Bodenham",
    author_email="deanbodenhampkgs@gmail.com",
    description="A collection of methods for computing the cdf of a weighted sum of chi-squared random variables.",
    long_description=long_description,
    long_description_content_type="A collection of moment-matching methods for computing the cumulative distribution function of a positively-weighted sum of chi-squared random variables. Methods include the Satterthwaite-Welch method, Hall-Buckley-Eagleson method and Wood's F method.",
    url="https://github.com/deanbodenham/momentchi2py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

