from setuptools import find_packages, setup

package_name = 'python_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anton',
    maintainer_email='a.bredenbeck@tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_pos_traj = python_controllers.example_pos_traj:main',
            'example_vel_traj = python_controllers.example_vel_traj:main',
            'triangle = python_controllers.triangle_traj:main',
            'figeight = python_controllers.figeight_traj:main',
            'given = python_controllers.given_traj:main',
            'pnp = python_controllers.pnp_traj:main',
            'path = python_controllers.traj:main',
            'stack = python_controllers.stack_traj:main',
            'vel = python_controllers.vel_traj:main',
        ],
    },
)
