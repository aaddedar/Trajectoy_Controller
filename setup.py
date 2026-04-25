from glob import glob

from setuptools import setup

package_name = 'control_car'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Simple ROS 2 package for car control (Python, rclpy).',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pure_persuit_node = control_car.pure_persuit_node:main',
        ],
    },
)
