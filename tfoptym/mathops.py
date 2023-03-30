"""want to configure a backend shim that allows us to switch to cupy, jax, etc.
taken from prysm.mathops with author's permission
"""
import numpy as np

class BackendShim:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)

_np = np
np = BackendShim(np)

def set_backend_to_numpy():
    """Convenience method to automatically configure tfoptym's backend to cupy."""
    import numpy as cp
    np._srcmodule = cp
    return

def set_backend_to_cupy():
    """Convenience method to automatically configure tfoptym's backend to cupy."""
    import cupy as cp
    np._srcmodule = cp
    return

def set_backend_to_jax():
    """Convenience method to automatically configure tfoptym's backend to cupy."""

    # Get the numpy module
    import jax.numpy as jnp

    # jax defaults to 32 bit but we need 64bit
    from jax.config import config
    config.update("jax_enable_x64", True)

    np._srcmodule = jnp
    return

def compute_eigenvalues_2x2(array):

  """ Computes the eigenvalues of a 2x2 matrix using a trick


  Parameters
  ----------
  array : numpy.ndarray
    a N x 2 x 2 array that we are computing the eigenvalues of

  Returns
  -------
  e1, e2 : floats of shape N
    The eigenvalues of the array
  """

  a = array[...,0,0]
  b = array[...,0,1]
  c = array[...,1,0]
  d = array[...,1,1]

  determinant = (a*d - b*c)
  mean_ondiag = (a+d)/2
  e1 = mean_ondiag + np.sqrt(mean_ondiag**2 - determinant)
  e2 = mean_ondiag - np.sqrt(mean_ondiag**2 - determinant)

  return e1,e2 