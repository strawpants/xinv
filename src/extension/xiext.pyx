# This file is part of the shxarray software which is licensed
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

#
# distutils: language = c++
# cython: profile=False
"""
**xiext** is xinv  binary Cython backend. 
Some of the heavy lifting is done using this the functions of this shared library.
This file groups cython functionality under a common shared library
"""

include "sinex.pyx" 
