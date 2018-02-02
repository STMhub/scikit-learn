# Synopsis: Projection onto L1 ball
# Author: Elvis Dohmatob <gmdopp@gmail.com>

from cython cimport floating

cdef void proj_l1(int n, floating *w, floating reg,
                  floating ajj) nogil
cdef void enet_projection(unsigned int m, floating *v, floating *out, floating radius,
                          floating l1_ratio) nogil

