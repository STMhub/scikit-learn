# Author: Elvis Dohmatob <gmdopp@gmail.com>
# License: BSD

from cython cimport floating

"""Projection unto l1 ball.
"""
cdef void proj_l1(int n, floating *w, floating reg,
                  floating ajj) nogil

