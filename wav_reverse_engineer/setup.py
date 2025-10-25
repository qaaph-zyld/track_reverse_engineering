from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wav-reverse-engineer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful tool for analyzing and reverse engineering WAV audio files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wav-reverse-engineer",
    packages=find_packages(),
    package_data={
        'audio_analyzer': ['*.py'],
    },
    install_requires=[
        'librosa>=0.10.0',
        'numpy>=1.20.0',
        'matplotlib>=3.5.0',
        'scipy>=1.7.0',
        'pydub>=0.25.1',
        'pretty_midi>=0.2.10',
        'madmom>=0.16.1',
        'scikit-learn>=1.0.0',
        'soundfile>=0.10.3',
        'python-rtmidi>=1.4.9',
    ],
    entry_points={
        'console_scripts': [
            'wav-reverse-engineer=cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
    ],
    python_requires='>=3.7',
)

# Add a simple __main__.py to allow running as a module
with open("__main__.py", "w") as f:
    f.write("from cli import main\n\nif __name__ == "__main__":\n    main()")
