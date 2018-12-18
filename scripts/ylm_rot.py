import numpy as np


def get_ylm_coeffs(inc=90, obl=0, alpha=0):
    cosi = np.cos(inc * np.pi / 180)
    sini = np.sin(inc * np.pi / 180)
    cosl = np.cos(obl * np.pi / 180)
    sinl = np.sin(obl * np.pi / 180)
    A = sini * cosl
    B = sini * sinl
    C = cosi
    return np.array([0,
        np.sqrt(3)*np.pi*B*(-A**2*alpha - B**2*alpha - C**2*alpha + 5)/15,
        0,
        np.sqrt(3)*np.pi*A*(-A**2*alpha - B**2*alpha - C**2*alpha + 5)/15,
        0,
        0,
        0,
        0,
        0,
        np.sqrt(70)*np.pi*B*alpha*(3*A**2 - B**2)/70,
        2*np.sqrt(105)*np.pi*C*alpha*(-A**2 + B**2)/105,
        np.sqrt(42)*np.pi*B*alpha*(A**2 + B**2 - 4*C**2)/210,
        0,
        np.sqrt(42)*np.pi*A*alpha*(A**2 + B**2 - 4*C**2)/210,
        4*np.sqrt(105)*np.pi*A*B*C*alpha/105,
        np.sqrt(70)*np.pi*A*alpha*(A**2 - 3*B**2)/70])