from .mathops import np
from .mathops import compute_eigenvalues_2x2

def compute_thin_films_byu(stack, aoi, wavelength, ambient_index=1, substrate_index=0.4 + 1j*7, polarization='s'):
    
    """
    Parameters
    ----------
    stack : list of tupes
        [(n1,d1),(n2,d2),...,(nM,dM)] representative of the thin film stack
    aoi : float
        angle of incidence in radians
    wavelength : float
        wavelength of light in meters
    ambient_index : float, optional
        index of refraction of the medium the optical system is immersed in. Defaults to 1
    substrate_index : float, optional
        index of refraction of the substrate the thin film stack is deposited on. Defaults to 1.5
    polarization : str, optional
        polarization state to compute the fresnel coefficients for. options are "s" and "p". Defaults to "s"
        
    Returns
    -------
    rtot,ttot : floats
        Effective fresnel reflection and transmission coefficients.
    """

    # Consider the incident media
    system_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    # Consider the terminating media
    aor = np.arcsin(ambient_index/substrate_index*np.sin(aoi))

    for layer in stack:

        ni = np.complex128(layer[0])
        di = np.complex128(layer[1])

        angle_in_film = np.arcsin(ambient_index/ni*np.sin(aoi))

        Beta = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength

        if polarization == 'p':
            newfilm = np.array([[np.cos(Beta), -1j*np.sin(Beta)*np.cos(angle_in_film)/ni],
                                [-1j*ni*np.sin(Beta)/np.cos(angle_in_film), np.cos(Beta)]])

        elif polarization == 's':
            newfilm = np.array([[np.cos(Beta), -1j*np.sin(Beta)/(np.cos(angle_in_film)* ni)],
                                [-1j*ni*np.sin(Beta)*np.cos(angle_in_film), np.cos(Beta)]])

        if newfilm.ndim > 2:
          newfilm = np.moveaxis(newfilm,-1,0)
          newfilm = np.moveaxis(newfilm,-1,0)

        system_matrix = system_matrix @ newfilm

    # Final matrix
    coeff = 1/(2*ambient_index*np.cos(aoi))

    if polarization == 'p':
        front_matrix = np.array([[np.full_like(di,ambient_index), np.full_like(di,np.cos(aoi))],
                                 [np.full_like(di,ambient_index), np.full_like(di,-np.cos(aoi))]])
        
        back_matrix = np.array([[np.full_like(di,np.cos(aor)),  np.full_like(di,0.)],
                                [np.full_like(di,substrate_index), np.full_like(di,0.)]])
        
        if front_matrix.ndim > 2:
          front_matrix = np.moveaxis(front_matrix,-1,0)
          front_matrix = np.moveaxis(front_matrix,-1,0)
          back_matrix = np.moveaxis(back_matrix,-1,0)
          back_matrix = np.moveaxis(back_matrix,-1,0)

    elif polarization == 's':
        front_matrix = np.array([[np.full_like(di,ambient_index*np.cos(aoi)), np.full_like(di,1.)],
                                 [np.full_like(di,ambient_index*np.cos(aoi)), np.full_like(di,-1.)]])
        back_matrix = np.array([[np.full_like(di,1), np.full_like(di,0.)],
                                [np.full_like(di,substrate_index*np.cos(aor)),np.full_like(di,0.)]])
        
        if front_matrix.ndim > 2:
          front_matrix = np.moveaxis(front_matrix,-1,0)
          front_matrix = np.moveaxis(front_matrix,-1,0)
          back_matrix = np.moveaxis(back_matrix,-1,0)
          back_matrix = np.moveaxis(back_matrix,-1,0)
    print(f'front {polarization}')
    print(front_matrix)
    print(f'stack {polarization}')
    print(system_matrix)
    print(f'back {polarization}')
    print(back_matrix)
    characteristic_matrix = coeff * front_matrix @ system_matrix @ back_matrix

    ttot = 1/characteristic_matrix[..., 0, 0]
    rtot = characteristic_matrix[..., 1, 0]/characteristic_matrix[...,0, 0]

    return rtot,ttot

def compute_diattenuation_and_retardance(jones,return_throughput=False):

  """Computes the polarization aberrations given a homogenous jones pupil

  Parameters
  ----------
  jones : numpy.ndarray
    jones matrix of an optical system. Should generally be complex of shape
    N x 2 x 2 

  Returns
  -------
  D,R : numpy.ndarray
    arrays of shape N containing the Diatennuation (D) in [unitless] and Retardance (R) in radians.
  """

  eval1,eval2 = compute_eigenvalues_2x2(jones)
  throughput = (np.abs(eval1)**2 + np.abs(eval2)**2)
  D = (np.abs(eval1)**2 - np.abs(eval2)**2) / throughput
  R = np.angle(eval1) - np.angle(eval2)

  if return_throughput:
    return D,R,throughput
  else:
    return D,R

def compute_total_jones(jones_osys,jones_thinfilm,jones_osys_after=None):

  """
  Parameters
  ----------
  jones_osys : numpy.ndarray
    jones pupil of the optical system we want to analyze. Should generally be complex of shape
    N x 2 x 2
  jones_thinfilm : numpy.ndarray
    jones pupil of the thin film stack that is being optimized. Should generally be complex of shape
    N x 2 x 2
  jones_osys_after : numpy.ndarray, optional
    jones pupil of the optical system after the thin film stack. This is relevant if the thin film
    stack is inside the optical system instead of external
  """

  if jones_osys_after is not None:
    return jones_osys_after @ jones_thinfilm @ jones_osys
  else:
    return jones_thinfilm @ jones_osys