from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fewshot',
    ext_modules=[
        CUDAExtension(
            name='fewshot_cuda', 
            sources = [
            'backproject_kernel.cu',
            'fewshot_layers.cpp'],
            include_dirs = ['/usr/local/include/eigen3', '/usr/local/include'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
