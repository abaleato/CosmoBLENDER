import numpy as np
import hmvec as hm
from . import tools as tls

class hm_framework:
    ''' Set the halo model parameters '''
    def __init__(self, massCut_Mvir = 5e15, m_min=2e13, m_max=5e16, nMasses=30, z_min=0.07, z_max=3, nZs=30, k_min = 1e-4, k_max=10, nks=1001, mass_function='sheth-torman', mdef='vir', cosmoParams=None):
        ''' Inputs:
                * massCut_Mvir = Virial masss cut in solar masses.
                * m_min = Minimum virial mass for the halo model calculation
                * m_max = Maximum virial mass for the halo model calculation (bot that massCut_Mvir will overide this)
                * nMasses = Integer. Number of steps in mass for the integrals
                * z_min = Minimum redshift for the halo model calc
                * z_max = Maximum redshift for the halo model calc
                * nZs = Integer. Number of steps in redshift for the integrals
                * k_min = Minimum k for the halo model calc
                * k_max = Maximum k for the halo model calc
                * nks = Integer. Number of steps in k for the integrals
                * mass_function = String. Halo mass function to use. Must be coded into hmvec
                * mdef = String. Mass definition. Must be defined in hmvec for the chosen mass_function
                * cosmoParams = Dictionary of cosmological parameters to initialised HaloModel hmvec object
        '''
        self.massCut = massCut_Mvir #Convert from M_vir (which is what Alex uses) to M_200 (which is what the
                                    # Tinker mass function in hmvec uses) using the relation from White 01.
        self.nMasses = nMasses
        zs = np.linspace(z_min,z_max,nZs) # redshifts
        ms = np.geomspace(m_min,m_max,nMasses) # masses
        ks = np.geomspace(k_min,k_max,nks) # wavenumbers
        self.T_CMB = 2.7255e6
        self.nZs = nZs

        self.hcos = hm.HaloModel(zs,ks,ms=ms,mass_function=mass_function,params=cosmoParams,mdef=mdef)
        self.hcos.add_battaglia_pres_profile("y",family="pres",xmax=5,nxs=40000)

        self.ms_rescaled = self.hcos.ms[...]/self.hcos.rho_matter_z(0)
        mMask = np.ones(nMasses)
        mMask[self.massCut<self.hcos.ms]=0

        # This is an adhoc fix for the large scales. Perhaps not needed here.
        # Essentially the one halo terms are flat on large scales, this means the as k->0 you are dominated by these
        # terms rather than the two halo term, which tends to P_lin (for the matter halo model)
        # The consistency term should just subtract this off.
        # I have removed this for now as i think it is likley subdomiant
        self.consistency =  np.trapz(self.hcos.nzm*self.hcos.bh*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)

    def get_tsz_bias(self, exp):
        ''' Calculate the tsz biases given an "experiment" object (defined in qest.py)'''
        hcos = self.hcos
        # FIXME: Hard-codDe QE normalisation for now
        norm_from_lenscov = np.load('/Users/antonbaleatolizancos/Projects/lensing_rec_biases/auxiliary_objects/N0_lmax3000_nlevt18_beam1arcmin.npy')
        norm = np.interp(exp.ls, np.arange(3001), norm_from_lenscov)

        nx = exp.lmax+1
        # The one and two halo bias terms -- these store the integrand to be integrated over z
        oneHalo_4pt = np.zeros([nx,self.nZs])+0j
        twoHalo_4pt = np.zeros([nx,self.nZs])+0j
        oneHalo_cross = np.zeros([nx,self.nZs])+0j
        twoHalo_cross = np.zeros([nx,self.nZs])+0j

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_4pt = np.zeros([nx,self.nMasses])+0j
            integrand_oneHalo_cross = np.zeros([nx,self.nMasses])+0j
            integrand_twoHalo_2g = np.zeros([nx,self.nMasses])+0j
            integrand_twoHalo_1g = np.zeros([nx,self.nMasses])+0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> self.massCut: continue
                y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.pk_profiles['y'][i,j]*(1-np.exp(-(hcos.ks/hcos.p['kstar_damping']))), ellmax=exp.lmax)
                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.uk_profiles['nfw'][i,j]*hcos.lensing_window(hcos.zs[i],1100.), ellmax=exp.lmax)
                kfft = kap*self.ms_rescaled[j]

                unnormalised_phi = exp.get_unnorm_TT_qe(np.arange(exp.lmax+1), y,y)

                #Normalize the reconstruction
                phi_estimate_cfft = unnormalised_phi * norm
                phi_estimate_cfft = np.nan_to_num(phi_estimate_cfft)

                # Accumulate the integrands
                integrand_oneHalo_cross[:,j] = phi_estimate_cfft*np.conjugate(kfft)*hcos.nzm[i,j]
                integrand_oneHalo_4pt[:,j] = phi_estimate_cfft*np.conjugate(phi_estimate_cfft) * hcos.nzm[i,j]
                integrand_twoHalo_1g[:,j] = np.conjugate(kfft)*hcos.nzm[i,j]*hcos.bh[i,j]
                integrand_twoHalo_2g[:,j] = phi_estimate_cfft*hcos.nzm[i,j]*hcos.bh[i,j]

            # Perform the m integrals
            oneHalo_4pt[:,i]=np.trapz(integrand_oneHalo_4pt,hcos.ms,axis=-1)
            oneHalo_cross[:,i]=np.trapz(integrand_oneHalo_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=exp.lmax)
            twoHalo_4pt[:,i]=np.trapz(integrand_twoHalo_2g,hcos.ms,axis=-1)**2 *pk
            tmpCorr =np.trapz(integrand_twoHalo_1g,hcos.ms,axis=-1)
            twoHalo_cross[:,i]=np.trapz(integrand_twoHalo_2g,hcos.ms,axis=-1)*(tmpCorr + hcos.lensing_window(hcos.zs,1100.)[i] - hcos.lensing_window(hcos.zs[i],1100.)*self.consistency[i])*pk#

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * exp.ls*(exp.ls+1) ))

        # Integrate over z
        exp.biases['tsz']['trispec']['1h'] = tls.scale_sz(exp.freq_GHz)**4 * self.T_CMB**4* np.trapz(oneHalo_4pt*hcos.comoving_radial_distance(hcos.zs)**-6*(hcos.h_of_z(hcos.zs)**3),hcos.zs,axis=-1)
        exp.biases['tsz']['trispec']['2h'] = tls.scale_sz(exp.freq_GHz)**4 * self.T_CMB**4* np.trapz(twoHalo_4pt*hcos.comoving_radial_distance(hcos.zs)**-6*(hcos.h_of_z(hcos.zs)**3),hcos.zs,axis=-1)
        exp.biases['tsz']['prim_bispec']['1h'] = 2*conversion_factor * tls.scale_sz(exp.freq_GHz)**2 * self.T_CMB**2* np.trapz(oneHalo_cross*1./hcos.comoving_radial_distance(hcos.zs)**4*(hcos.h_of_z(hcos.zs)**2),hcos.zs,axis=-1)
        exp.biases['tsz']['prim_bispec']['2h'] = 2*conversion_factor * tls.scale_sz(exp.freq_GHz)**2 * self.T_CMB**2* np.trapz(twoHalo_cross*1./hcos.comoving_radial_distance(hcos.zs)**4*(hcos.h_of_z(hcos.zs)**2),hcos.zs,axis=-1)
        return
