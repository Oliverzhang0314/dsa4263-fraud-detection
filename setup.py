from setuptools import setup, find_packages

setup(
   name="fraud",
   version="0.1.0",
   license="MIT",
   description="Module to build classification model for the purpose of Malicious URL detection",
   author="Yi Fu ZHU",
   author_email="zhuyifu1998@gmail.com",
   url="https://github.com/Oliverzhang0314/dsa4263-fraud-detection",
   packages=find_packages(include=["fraud"]),
   install_requires=[
       "numpy==1.26.4",
       "pandas==2.2.1",
       "matplotlib==3.8.3",
       "seaborn==0.13.2",
       "pillow==10.2.0",
       "scikit-learn==1.3.0",
       "lazypredict==0.2.12",
       "shap==0.45.0",
       "pycebox==0.0.1",
       "kaggle",
       "opendatasets",
    ],
   python_requires=">=3.12",
)
