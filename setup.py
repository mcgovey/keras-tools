import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-rnn-tools", # Replace with your own username
    version="0.0.1",
    author="Kevin McGovern",
    author_email="author@example.com",
    description="A package that encapsulates commonly used tools for creating RNN models in Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcgovey/keras-lstm-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)