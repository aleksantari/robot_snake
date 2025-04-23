from setuptools import find_packages, setup
from glob import glob

package_name = 'snake_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # standard ament index registration
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        # your package.xml
        (f'share/{package_name}', ['package.xml']),
        # install everything under models/ into share/<pkg>/models/
        (f'share/{package_name}/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aleksantari',
    maintainer_email='aleksantari@gmail.com',
    description='Snake segmentation using YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = snake_segmentation.snake_segmentation_node:main',
        ],
    },
)
