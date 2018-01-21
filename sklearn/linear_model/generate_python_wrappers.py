# Synopsis: generate wrappers to cython codes for testing purposes
# Author: Elvis Dohmatob <gmdopp@gmail.com>

import sys
import glob


def main(verbose=0):
    with open("python_wrappers.pyx", 'w') as fd:
        def write_stuff(stuff):
            fd.write("%s\n" % stuff)
            if verbose:
                print(stuff)

        write_stuff("# Authomatically generated by %s\n" % sys.argv[0])
        write_stuff("\n### Prox / proj wrappers")
        write_stuff("cimport numpy as np")
        write_stuff("from types cimport floating, complexing")
        proxes = []
        for module in sorted(glob.glob("pro[jx]_*.pyx")):
            prox = module.split(".pyx")[0]
            write_stuff("from %s cimport %s" % (prox, prox))
            proxes.append(prox)
        write_stuff("")

        for prox in proxes:
            write_stuff(
                """
def _py_%s(np.ndarray[complexing, ndim=1] w, floating reg, floating ajj=1.):
                %s(len(w), &w[0], reg, ajj)\n""" % (prox, prox))

        # BLAS wrappers
        write_stuff("\n### BLAS wrappers")
        write_stuff("cimport numpy as np")
        write_stuff("from types cimport complexing")
        apis = []
        for api in ["fused_dotc", "fused_dotu"]:
            write_stuff("from blas_api cimport %s" % api)
            apis.append(api)
        write_stuff("")
        for api in apis:
            write_stuff(
"""
def _py_%s(np.ndarray[complexing, ndim=1] X,
                   np.ndarray[complexing, ndim=1] Y):
    return %s(len(X), &X[0], 1, &Y[0], 1)\n""" % (api, api))

        write_stuff("\n### More wrappers")
        write_stuff("from blas_api cimport fused_asum")
        write_stuff("")
        write_stuff("""
def _py_fused_asum(np.ndarray[complexing, ndim=1] w):
    return fused_asum(len(w),
                      &w[0],
                      1)""")


if __name__ == "__main__":
    main(verbose=1)