from types cimport floating, complexing


cdef inline floating fmax(floating x, floating y) nogil:
    return y if x < y else x


cdef double abs_max(int n, complexing* a) nogil:
    """np.max(np.abs(a))"""
    if complexing is float or complexing is double:
        abs = fabs
    elif complexing is double:
        abs = cabs
    else:
        abs = cabsf

    cdef int i
    cdef double m = abs(a[0])
    cdef double d
    for i in range(1, n):
        d = abs(a[i])
        if d > m:
            m = d
    return m


cdef floating fmax_arr(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef double diff_abs_max(int n, complexing* a, complexing* b) nogil:
    """np.max(np.abs(a - b))"""
    if complexing is double or complexing is float:
        abs = fabs
    elif complexing is complex:
        abs = cabs
    else:
        abs = cabsf
    cdef int i
    cdef double m = abs(a[0] - b[0])
    cdef double d
    for i in range(1, n):
        d = abs(a[i] - b[i])
        if d > m:
            m = d
    return m


cdef inline void relu(int n, floating *x) nogil:
    cdef int inc
    for inc in range(n):
        if x[inc] < 0.:
            x[inc] = 0.
