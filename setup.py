from setuptools import setup

setup(name='web_crawler',
      version='0.0.1',
      description='Python utility to scrap data from websites',
      url='http://github.com/fivosts/pinkySpeaker',
      author='Foivos Tsimpourlas',
      author_email='fivos_ts@hotmail.com',
      license='MIT',
      packages=['web'],
      install_dir="~/.local/lib/python3.7/dist-packages/",
      install_requires=[
          'scrapy'
      ],
      zip_safe=True)
