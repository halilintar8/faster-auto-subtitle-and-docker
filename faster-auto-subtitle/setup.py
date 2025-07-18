from setuptools import setup, find_packages

setup(
    version="1.3",
    name="faster_auto_subtitle",
    packages=find_packages(),
    py_modules=["auto_subtitle"],
    author="Sergey Chernyaev",
    install_requires=[
        'faster-whisper==1.0.3',
        'tqdm==4.66.4',
        'ffmpeg-python==0.2.0',
        'langcodes==3.3.0',
        'transformers>=4.4,<5',
        'torch>=1.6.0,<=2.3.1',
        'numpy>=1.24.2,<=1.26.4',
        'nltk<=3.8.1',
        'sentencepiece<=0.2.0',
        'protobuf<=5.27.2',
        'huggingface_hub<=0.23.5',
    ],
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        'console_scripts': ['faster_auto_subtitle=auto_subtitle.cli:main'],
    },
    include_package_data=True,
)