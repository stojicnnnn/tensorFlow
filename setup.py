# setup.py

from setuptools import setup, find_packages

setup(
    name='industry_picking',
    version='0.1.0',
    author='Nikola Stojic',
    author_email='snikola.stojic@gmail.com',
    description='Set of packages helping with point cloud acquisition and processing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/stojicnnnn/3DVision',

    # Automatically find all packages in the project
    packages=find_packages(where='industry_picking'),
    package_dir={'': 'industry_picking'},

    # List all project dependencies
    install_requires=[
        'rerun-sdk',
        'numpy',
        'open3d',
        'opencv-python',
        'requests',
        'matplotlib',
        'scipy',
        'xarm-python-sdk',
        'pyrealsense2',
        'transforms3d',
    ],

    # Create a command-line executable
    entry_points={
        'console_scripts': [
            'awesome-tool = my_awesome_tool.main:cli_entry_point',
        ],
    },

    python_requires='>=3.11.2',
)