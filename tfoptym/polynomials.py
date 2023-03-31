from .mathops import np

def gen_positive_polynomial_basis(nterms=3,npix=256,center=[0,0]):
    """generates a basis that takes the form \sum \sum x^n y^m where n,m are even numbers

    Parameters
    ----------
    nterms : int, optional
        number of nm pairs to generate, by default 3
    npix : int, optional
        number of pixels pixels on the side of the array, by default 256

    Returns
    -------
    basis : list of np arrays
        shapes of length nterms+1 
    """

    # configure positions from 0 to 1
    x = np.linspace(-1,1,npix,dtype=np.complex128)
    x,y = np.meshgrid(x,x)
    x = x-center[0]
    y = y-center[1]

    nms = np.arange(0,nterms,2)
    basis = []
    for n in nms:
        for m in nms:
            fun = x**n + y**m
            fun = fun / np.max(fun) # normalize to 1 at peak
            basis.append(fun)
    
    return basis