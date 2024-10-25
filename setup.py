from setuptools import find_packages,setup
from typing import List


hypen_e_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    # read the requirement.txt line by line
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines() # read lines from the file
        requirements=[req.replace('\n','') for req in requirements] # when we go to the next line \n is added so to remove the \n

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

setup(
    author="om",
    author_email="omkuman@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
# to run this file and install packages "install -r requirements.txt"