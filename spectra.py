'''
Files taken from quicklens' spec.py and made to work on Python 3.
'''
import numpy as np
import os
import glob

def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0, lmax+1)
    return np.exp( -(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.) )

def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
         * noise_uK_arcmin = map noise level in uK.arcmin
         * fwhm_arcmin     = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax            = maximum multipole.
    """
    return (noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax)**2

def get_camb_scalcl(fname=None, prefix=None, lmax=None):
    """ loads and returns a "scalar Cls" file produced by CAMB (camb.info).

         * (optional) fname  = file name to load.
         * (optional) prefix = directory in quicklens/data/cl directory to pull the *_scalCls.dat from. defaults to 'planck_wp_highL' (only used if fname==None).
         * (optional) lmax   = maximum multipole to load (all multipoles in file will be loaded by default).
    """
    if fname is None:
        basedir = os.path.dirname(__file__)
    
        if prefix is None:
            prefix = "planck_wp_highL"
        fname = basedir + "/data/cl/" + prefix + "/*_scalCls.dat"
    else:
        assert( prefix is None )
    print(fname)
    tf = glob.glob( fname )
    assert(len(tf) == 1),"No filename matching {0} found!".format(fname)
    
    return camb_clfile( tf[0], lmax=lmax)
    
def get_camb_lensedcl(fname=None, prefix=None, lmax=None):
    """ loads and returns a "lensed Cls" file produced by CAMB (camb.info).

         * (optional) fname  = file name to load.
         * (optional) prefix = directory in quicklens/data/cl directory to pull the *_lensedCls.dat from. defaults to 'planck_wp_highL' (only used if fname==None).
         * (optional) lmax   = maximum multipole to load (all multipoles in file will be loaded by default).
    """
    if fname is None:
        basedir = '/Users/antonbaleatolizancos/Software/Quicklens-with-fixes/quicklens/data' #os.path.dirname(__file__)

        if prefix is None:
            prefix = "planck_wp_highL"
        fname = basedir + "/data/cl/" + prefix + "/*_lensedCls.dat"

    tf = glob.glob( fname )
    assert(len(tf) == 1)
    return camb_clfile( tf[0], lmax=lmax )

class camb_clfile(object):
    """ class to load and store Cls from the output files produced by CAMB. """
    def __init__(self, tfname, lmax=None):
        """ load Cls.
             * tfname           = file name to load from.
             * (optional) lmax  = maximum multipole to load (all multipoles in file will be loaded by default).
        """
        tarray = np.loadtxt(tfname)
        lmin   = tarray[0, 0]
        assert(int(lmin)==lmin)
        lmin = int(lmin)

        if lmax is None:
            lmax = np.shape(tarray)[0]-lmin+1
            if lmax > 10000:
                lmax = 10000
            else:
                assert(tarray[-1, 0] == lmax)
        assert( (np.shape(tarray)[0]+1) >= lmax )

        ncol = np.shape(tarray)[1]
        ell  = np.arange(lmin, lmax+1, dtype=np.float)

        self.lmax = lmax
        self.ls   = np.concatenate( [ np.arange(0, lmin), ell ] )
        if ncol == 5:                                                                            # _lensedCls
            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.) ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.) ] )
            self.clbb = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.) ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]*2.*np.pi/ell/(ell+1.) ] )

        elif ncol == 6:                                                                          # _scalCls
            tcmb  = 2.726*1e6 #uK

            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.) ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.) ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.) ] )
            self.clpp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]/ell**4/tcmb**2 ] )
            self.cltp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),5]/ell**3/tcmb ] )
