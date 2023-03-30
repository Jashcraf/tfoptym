from .mathops import np
from .mathops import compute_eigenvalues_2x2

def compute_thin_films_macleod(stack, aoi, wavelength, ambient_index=1, substrate_index=1.5):
    """
    Parameters
    ----------

    stack : list of tuples
        list composed of elements containing the index (n) and thickness (t) in meters, ordered like
        stack = [(n0,t0),(n1,t1)...,(nN,tN)]. nN and tN are of the same shape, but can be any shape.

    aoi : float
        angle of incidence in radians on the thin film stack

    wavelength: float
        wavelength to comput the reflection coefficients for in meters

    """

    # Consider the incident media
    system_matrix_s = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    system_matrix_p = np.copy(system_matrix_s)

    # do some pre-computing
    aor = np.arcsin(ambient_index/substrate_index*np.sin(aoi))
    cosAOI = np.cos(aoi)
    cosAOR = np.cos(aor)
    sinAOI = np.sin(aoi)
    n0sinAOI = ambient_index*sinAOI

    eta_ambient_s = ambient_index*cosAOI
    eta_ambient_p = ambient_index/cosAOI

    # Consider the terminating media
    eta_substrate_s = substrate_index*cosAOR
    eta_substrate_p = substrate_index/cosAOR

    for layer in stack:

        ni = layer[0]
        di = layer[1]

        angle_in_film = np.arcsin(n0sinAOI/ni)

        phase_thickness = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength
        eta_film_s = ni*np.cos(angle_in_film)
        eta_film_p = ni/np.cos(angle_in_film)

        cosB = np.cos(phase_thickness)
        sinB = np.sin(phase_thickness)

        newfilm_s = np.array([[cosB, 1j*sinB/eta_film_s],
                              [1j*eta_film_s*sinB, cosB]])

        newfilm_p = np.array([[cosB, 1j*sinB/eta_film_p],
                              [1j*eta_film_p*sinB, cosB]])
        
        if newfilm_s.ndim > 2:
            newfilm_p = np.moveaxis(newfilm_p, -1, 0)
        if newfilm_p.ndim > 2:
            newfilm_p = np.moveaxis(newfilm_p,-1,0)

        system_matrix_s = system_matrix_s @ newfilm_s
        system_matrix_p = system_matrix_p @ newfilm_p

    # Computes the s-vector
    s_vector_substrate = np.array([np.full_like(eta_substrate_s, 1), eta_substrate_s])
    if s_vector_substrate.ndim >2:
        s_vector_substrate = np.moveaxis(s_vector_substrate, -1, 0)
    s_vector_substrate = s_vector_substrate[..., np.newaxis]
    s_vector = system_matrix_s @ s_vector_substrate
    Bs, Cs = s_vector[..., 0, 0], s_vector[..., 1, 0]

    # Computes the p-vector
    p_vector_substrate = np.array(
        [np.full_like(eta_substrate_p, 1), eta_substrate_p])
    if p_vector_substrate.ndim > 2:
        p_vector_substrate = np.moveaxis(p_vector_substrate, -1, 0)
    p_vector_substrate = p_vector_substrate[..., np.newaxis]
    p_vector = system_matrix_p @ p_vector_substrate
    Bp, Cp = p_vector[..., 0, 0], p_vector[..., 1, 0]

    Ys = Cs/Bs
    Yp = Cp/Bp

    rs = (eta_ambient_s - Ys)/(eta_ambient_s + Ys)
    rp = (eta_ambient_p - Yp)/(eta_ambient_p + Yp)

    # phase change on reflection from Chipman, absent from Macleod ch 2
    # Implementing results in spiky phase v.s. wavelength
    # phi_s = -np.arctan(np.imag(eta_substrate_s*(Bs*np.conj(Cs)-np.conj(Bs)*Cs))/(eta_substrate_s**2 * Bs*np.conj(Bs) - Cs * np.conj(Cs)))
    # phi_p = -np.arctan(np.imag(eta_substrate_p*(Bp*np.conj(Cp)-np.conj(Bp)*Cp))/(eta_substrate_p**2 * Bp*np.conj(Bp) - Cp * np.conj(Cp)))

    # rs *= np.exp(-1j*phi_s)
    # rp *= np.exp(-1j*phi_p)

    return rs, rp

def compute_thin_films_byu(stack, aoi, wavelength, ambient_index=1, substrate_index=1.5, polarization='s'):

    # Consider the incident media
    system_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)

    # Consider the terminating media
    aor = np.arcsin(ambient_index/substrate_index*np.sin(aoi))
    cosAOI = np.cos(aoi)
    sinAOI = np.sin(aoi)
    ncosAOI = ambient_index * cosAOI

    for layer in stack:

        ni = layer[0]
        di = layer[1]

        angle_in_film = np.arcsin(ambient_index/ni*sinAOI)

        Beta = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength

        cosB = np.cos(Beta)
        sinB = np.sin(Beta)
        cosT = np.cos(angle_in_film)

        if polarization == 'p':
            newfilm = np.array([[cosB, -1j*sinB*cosT/ni],
                                [-1j*ni*sinB/cosT, cosB]])

        elif polarization == 's':
            newfilm = np.array([[cosB, -1j*sinB/(cosT* ni)],
                                [-1j*ni*sinB*cosT, cosB]])
        if newfilm.ndim > 2: 
            newfilm = np.moveaxis(newfilm, -1, 0)

        system_matrix = system_matrix @ newfilm

    # Final matrix
    coeff = 1/(2*ncosAOI)

    if polarization == 'p':
        front_matrix = np.array([[ambient_index, cosAOI],
                                 [ambient_index, -cosAOI]])
        back_matrix = np.array([[np.cos(aor),  0],
                                [substrate_index, 0]])

    elif polarization == 's':
        front_matrix = np.array([[ncosAOI, 1],
                                 [ncosAOI, -1]])
        back_matrix = np.array([[1,  0],
                                [substrate_index*np.cos(aor),  0]])
    
    if back_matrix.ndim > 2:
        back_matrix = np.moveaxis(back_matrix,-1,0)

    characteristic_matrix = coeff * (front_matrix @ system_matrix @ back_matrix)

    ttot = 1/characteristic_matrix[..., 0, 0]
    rtot = characteristic_matrix[..., 1, 0]/characteristic_matrix[..., 0, 0]

    return rtot, ttot

def _compute_thin_films_byu(stack, aoi, wavelength, ambient_index=1, substrate_index=0.4 + 1j*7, polarization='s'):
    
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
    aor = np.lib.scimath.arcsin(ambient_index/substrate_index*np.sin(aoi))

    for layer in stack:

        ni = layer[0]
        di = layer[1]

        angle_in_film = np.lib.scimath.arcsin(ambient_index/ni*np.sin(aoi))

        Beta = 2 * np.pi * ni * di * np.cos(angle_in_film) / wavelength

        if polarization == 'p':
            newfilm = np.array([[np.cos(Beta), -1j*np.sin(Beta)*np.cos(angle_in_film)/ni],
                                [-1j*ni*np.sin(Beta)/np.cos(angle_in_film), np.cos(Beta)]])

        elif polarization == 's':
            newfilm = np.array([[np.cos(Beta), -1j*np.sin(Beta)/(np.cos(angle_in_film)* ni)],
                                [-1j*ni*np.sin(Beta)*np.cos(angle_in_film), np.cos(Beta)]])

        # if newfilm.ndim > 2:
        #   print(newfilm.shape)
        #   newfilm = np.moveaxis(newfilm,-1,0)
        #   print(newfilm.shape)

        system_matrix = system_matrix @ newfilm

    # Final matrix
    coeff = 1/(2*ambient_index*np.cos(aoi))

    if polarization == 'p':
        front_matrix = np.array([[ambient_index, np.cos(aoi)],
                                 [ambient_index, -np.cos(aoi)]])
        
        back_matrix = np.array([[np.cos(aor),  0.],
                                [substrate_index, 0.]])

    elif polarization == 's':
        front_matrix = np.array([[ambient_index*np.cos(aoi), 1.],
                                 [ambient_index*np.cos(aoi), -1.]])
        back_matrix = np.array([[1, 0.],
                                [substrate_index*np.cos(aor),0.]])
        
    # if front_matrix.ndim > 2:
    #   front_matrix = np.moveaxis(front_matrix,-1,0)
    #   back_matrix = np.moveaxis(back_matrix,-1,0)

    print(f'front {polarization}')
    print(front_matrix)
    print(f'stack {polarization}')
    print(system_matrix)
    print(f'back {polarization}')
    print(back_matrix)
    characteristic_matrix = coeff * (front_matrix @ system_matrix @ back_matrix)

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