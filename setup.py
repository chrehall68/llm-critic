import setuptools

setuptools.setup(
    name="llm_critic",
    packages=setuptools.find_packages(exclude=["scripts"]),
    requires=open("requirements.txt").read().split("\n"),
)
