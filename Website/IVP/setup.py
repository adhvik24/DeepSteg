# This code logic is from github we have seperated it from the github https://github.com/daniellerch/hstego and we have modified it to work with our project. We extracted the Juniward part from this so to encode images with J-UNIWARD .We have appropriately commented the code logic so that we can understand it.

import os
from setuptools import setup, Extension

# m_jpg is the extension module for the jpeg_toolbox library
# jpeg_toolbox library is a C library, so we need to compile it with gcc
# the library is used to read and write JPEG images
# the library is compiled with -O3 flag to optimize the code
# this library is used by the stc_embed_c and stc_extract_c modules
# it is helpful in many ways, for example, it allows us to use the same code for embedding and extracting messages
m_jpg = Extension('hstego_jpeg_toolbox_extension', 
                  sources = ['src/jpeg_toolbox_extension.c'], 
                  libraries = ['jpeg'])

# m_stc is the extension module for the stc library
# stc library is a C++ library, so we need to compile it with g++
# the library is compiled with C++11 standard
# it is used to embed and extract messages from images
# the library is compiled with -Wno-narrowing flag to suppress narrowing conversion warnings
# the library is compiled with -O3 flag to optimize the code
# this library is used by the stc_embed_c and stc_extract_c modules
# it is helpful in many ways, for example, it allows us to use the same code for embedding and extracting messages
m_stc = Extension('hstego_stc_extension', 
                  include_dirs = ['src/'],
                  sources = ['src/common.cpp',
                             'src/stc_embed_c.cpp',
                             'src/stc_extract_c.cpp',
                             'src/stc_interface.cpp',
                             'src/stc_ml_c.cpp'],
                  extra_compile_args = ['-std=c++11', '-Wno-narrowing'],
                  )


setup(ext_modules = [m_jpg, m_stc])