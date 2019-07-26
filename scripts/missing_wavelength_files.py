import numpy as np
from astropy.io import fits
import os
import glob

def missing_wavelength_files(filelist):
    missing_files = []
    for f in filelist:
        path = f[0:str.rfind(f,'/')+1]
        sp = fits.open(f)
        header = sp[0].header
        wave_file = header['HIERARCH ESO DRS CAL TH FILE']
        if os.path.isfile(path+wave_file):
            continue
        else:
            missing_files = np.append(missing_files, wave_file)
            
    return np.unique(missing_files)
    
    
if __name__ == '__main__':
    filelist = glob.glob('/Users/mbedell/python/wobble/data/51peg/HARPS*ccf_G2_A.fits')
    np.savetxt('missing_files.txt', missing_files, fmt='%s')