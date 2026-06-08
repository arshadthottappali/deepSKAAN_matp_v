from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_text = f.read()

setup(
    name='deepSKAN',
    version='2.0.0',
    description=(
        'Automated kinetic model classification for transient absorption spectroscopy '
        'using a deep residual CNN, followed by Global and Target Analysis fitting.'
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Philipp Kollenz',
    author_email='p.kollenz@stud.uni-heidelberg.de',
    license=license_text,
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.0',
        'numpy>=1.24',
        'scipy>=1.10',
        'matplotlib>=3.7',
        'networkx>=3.0',
        'tqdm>=4.65',
        'h5py>=3.8',
    ],
    packages=find_packages(exclude=('tests', 'docs', 'examples', 'deepGTA')),
    entry_points={
        'console_scripts': [
            'deepskan-train    = train:train',
            'deepskan-analyze  = analyze:analyze',
            'deepskan-evaluate = evaluate:main',
            'deepskan-generate = generate_dataset:generate',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
