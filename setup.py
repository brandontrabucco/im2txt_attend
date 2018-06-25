from setuptools import setup

setup(name='im2txt_attend',
      version='0.1',
      description='An image captioning framework using tensorflow',
      url='http://github.com/brandontrabucco/im2txt_attend',
      author='Brandon Trabucco',
      author_email='brandon@btrabucco.com',
      license='MIT',
      packages=['im2txt_attend', 'im2txt_attend.inference_utils', 'im2txt_attend.ops'],
      zip_safe=False)
