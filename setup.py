import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='resonances',
    version='0.2.1',
    author='Evgeny Smirnov',
    author_email='smirik@gmail.com',
    description='Identification of mean-motion resonances',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/smirik/resonances',
    project_urls={
        'Bug Tracker': 'https://github.com/smirik/resonances/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'resonances'},
    packages=setuptools.find_packages(where='resonances'),
    python_requires='>=3.7.1',
    keywords='astronomy celestial-mechanics nbody mean-motion-resonance resonance mmr',
)
