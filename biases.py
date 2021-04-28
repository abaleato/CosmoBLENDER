import numpy as np
import hmvec as hm
import tools as tls
import second_bispec_bias_stuff as sbbs #FIXME:remove this when the secondary bispectrum bias is properly incorporated
import quicklens as ql

class hm_framework:
    """ Set the halo model parameters """
    def __init__(self, lmax_out=3000, m_min=2e13, m_max=5e16, nMasses=30, z_min=0.07, z_max=3, nZs=30, k_min = 1e-4,\
                 k_max=10, nks=1001, mass_function='sheth-torman', mdef='vir', cib_model='planck13', cosmoParams=None, xmax=5, nxs=40000):
        """ Inputs:
                * lmax_out = int. Maximum multipole at which to return the lensing reconstruction
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
                * cib_model = CIB halo model and fit params. Either 'planck13' or 'vierro' (the latter after Viero et al 13.)
                * cosmoParams = Dictionary of cosmological parameters to initialised HaloModel hmvec object
                * xmax = Float. Electron pressure profile integral xmax (see further docs at hmvec.add_nfw_profile() )
                * nxs = Integer. Electron pressure profile integral number of x's
        """
        self.lmax_out = lmax_out
        self.nMasses = nMasses
        self.m_min = m_min
        self.m_max = m_max
        self.z_min = z_min
        self.z_max = z_max
        self.nZs = nZs
        self.mass_function = mass_function
        self.mdef = mdef
        zs = np.linspace(z_min,z_max,nZs) # redshifts
        ms = np.geomspace(m_min,m_max,nMasses) # masses
        ks = np.geomspace(k_min,k_max,nks) # wavenumbers
        self.T_CMB = 2.7255e6
        self.nZs = nZs
        self.nxs = nxs
        self.xmax = xmax
        self.cosmoParams = cosmoParams

        self.hcos = hm.HaloModel(zs,ks,ms=ms,mass_function=mass_function,params=cosmoParams,mdef=mdef)
        self.hcos.add_battaglia_pres_profile("y",family="pres",xmax=xmax,nxs=nxs)
        self.hcos.set_cibParams(cib_model)

        self.ms_rescaled = self.hcos.ms[...]/self.hcos.rho_matter_z(0)

    def __str__(self):
        """ Print out halo model calculator properties """
        m_min = '{:.2e}'.format(self.m_min)
        m_max = '{:.2e}'.format(self.m_max)
        z_min = '{:.2f}'.format(self.z_min)
        z_max = '{:.2f}'.format(self.z_max)

        return 'M_min: ' + m_min + '  M_max: ' + m_max + '  n_Masses: '+ str(self.nMasses) + '\n' + '  z_min: ' + z_min + '  z_max: ' + z_max + '  n_zs: ' + str(self.nZs) +  '\n' +'  Mass function: ' + self.mass_function + '  Mass definition: ' + self.mdef

    def get_consistency(self, exp):
        """
        Calculate consistency relation for 2-halo term given some mass cut
        Input:
            * exp = a qest.experiment object
        """
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0

        # This is an adhoc fix for the large scales. Perhaps not needed here.
        # Essentially the one halo terms are flat on large scales, this means the as k->0 you are dominated by these
        # terms rather than the two halo term, which tends to P_lin (for the matter halo model)
        # The consistency term should just subtract this off.
        # I have removed this for now as i think it is likley subdomiant
        self.consistency =  np.trapz(self.hcos.nzm*self.hcos.bh*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)

    def get_tsz_bias(self, exp, fftlog_way=True, get_secondary_bispec_bias=False, bin_width_out=30, bin_width_out_second_bispec_bias=1000):
        """
        Calculate the tsz biases given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) bin_width_out_second_bispec_bias = int. Bin width of the output secondary bispectrum bias
        """
        hcos = self.hcos
        self.get_consistency(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        if get_secondary_bispec_bias:
            lbins_second_bispec_bias = np.arange(1,self.lmax_out+1,bin_width_out_second_bispec_bias)
            conversion_factor_second_bispec_bias = np.nan_to_num(1 / (0.5 * lbins_second_bispec_bias * (lbins_second_bispec_bias + 1)))

        nx = self.lmax_out+1 if fftlog_way else exp.pix.nx

        # The one and two halo bias terms -- these store the integrand to be integrated over z.
        # Dimensions depend on method
        oneHalo_4pt = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoHalo_4pt = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        oneHalo_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoHalo_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        if get_secondary_bispec_bias:
            oneHalo_second_bispec = np.zeros([len(lbins_second_bispec_bias),self.nZs])+0j

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_4pt = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            integrand_oneHalo_cross =np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            integrand_twoHalo_2g = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            integrand_twoHalo_1g = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            if get_secondary_bispec_bias:
                integrand_oneHalo_second_bispec = np.zeros([len(lbins_second_bispec_bias),self.nMasses])+0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.pk_profiles['y'][i,j]\
                                 *(1-np.exp(-(hcos.ks/hcos.p['kstar_damping']))), ellmax=exp.lmax)
                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.uk_profiles['nfw'][i,j]\
                                   *hcos.lensing_window(hcos.zs[i],1100.), ellmax=self.lmax_out)
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                phi_estimate_cfft = exp.get_TT_qe(fftlog_way, ells_out, y,y)

                # Accumulate the integrands
                integrand_oneHalo_cross[...,j] = phi_estimate_cfft*np.conjugate(kfft)*hcos.nzm[i,j]
                integrand_oneHalo_4pt[...,j] = phi_estimate_cfft*np.conjugate(phi_estimate_cfft) * hcos.nzm[i,j]
                integrand_twoHalo_1g[...,j] = np.conjugate(kfft)*hcos.nzm[i,j]*hcos.bh[i,j]
                integrand_twoHalo_2g[...,j] = phi_estimate_cfft*hcos.nzm[i,j]*hcos.bh[i,j]

                if get_secondary_bispec_bias:
                    # Temporary secondary bispectrum bias stuff
                    # The part with the nested lensing reconstructions
                    # FIXME: currently the following is incompatible with fftlog_way=False (bc of kfft, conversion_factor, etc)
                    secondary_bispec_bias_reconstructions = sbbs.get_secondary_bispec_bias(lbins_second_bispec_bias, exp, y, kfft)
                    integrand_oneHalo_second_bispec[..., j] = secondary_bispec_bias_reconstructions
                    # FIXME:add the 2-halo term. Should be easy.

            # Perform the m integrals
            oneHalo_4pt[...,i]=np.trapz(integrand_oneHalo_4pt,hcos.ms,axis=-1)
            oneHalo_cross[...,i]=np.trapz(integrand_oneHalo_cross,hcos.ms,axis=-1)
            if get_secondary_bispec_bias:
                oneHalo_second_bispec[...,i]=np.trapz(integrand_oneHalo_second_bispec,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            twoHalo_4pt[...,i]=np.trapz(integrand_twoHalo_2g,hcos.ms,axis=-1)**2 *pk
            tmpCorr =np.trapz(integrand_twoHalo_1g,hcos.ms,axis=-1)
            #FIXME: do we need to apply consistency condition to integral over fg profiles too? So far only kappa part
            twoHalo_cross[...,i]=np.trapz(integrand_twoHalo_2g,hcos.ms,axis=-1)\
                                 *(tmpCorr + hcos.lensing_window(hcos.zs,1100.)[i]\
                                   - hcos.lensing_window(hcos.zs[i],1100.)*self.consistency[i])*pk

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # Integrate over z
        exp.biases['tsz']['trispec']['1h'] = tls.scale_sz(exp.freq_GHz)**4 * self.T_CMB**4 \
                                             * np.trapz(oneHalo_4pt*hcos.comoving_radial_distance(hcos.zs)**-6\
                                                        *(hcos.h_of_z(hcos.zs)**3),hcos.zs,axis=-1)
        exp.biases['tsz']['trispec']['2h'] = tls.scale_sz(exp.freq_GHz)**4 * self.T_CMB**4 \
                                             * np.trapz(twoHalo_4pt*hcos.comoving_radial_distance(hcos.zs)**-6\
                                                        *(hcos.h_of_z(hcos.zs)**3),hcos.zs,axis=-1)
        exp.biases['tsz']['prim_bispec']['1h'] = 2*conversion_factor * tls.scale_sz(exp.freq_GHz)**2 * self.T_CMB**2 \
                                                 * np.trapz(oneHalo_cross*1./hcos.comoving_radial_distance(hcos.zs)**4\
                                                            *(hcos.h_of_z(hcos.zs)**2),hcos.zs,axis=-1)
        exp.biases['tsz']['prim_bispec']['2h'] = 2*conversion_factor * tls.scale_sz(exp.freq_GHz)**2 * self.T_CMB**2 \
                                                 * np.trapz(twoHalo_cross*1./hcos.comoving_radial_distance(hcos.zs)**4\
                                                            *(hcos.h_of_z(hcos.zs)**2),hcos.zs,axis=-1)
        if get_secondary_bispec_bias:
        # FIXME: check the prefactors here
            exp.biases['tsz']['second_bispec']['1h'] = conversion_factor_second_bispec_bias * tls.scale_sz(
                exp.freq_GHz) ** 2 * self.T_CMB ** 2 * np.trapz( oneHalo_second_bispec * 1.\
                                                                 / hcos.comoving_radial_distance(hcos.zs) ** 4\
                                                                 * (hcos.h_of_z(hcos.zs) ** 2), hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = lbins_second_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['tsz']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_tsz_ps(self, exp):
        """
        Calculate the tSZ power spectrum
        Input:
            * exp = a qest.experiment object
        """
        hcos = self.hcos

        # Output ells
        ells_out = np.arange(self.lmax_out+1)

        nx = self.lmax_out+1

        # The one and two halo bias terms -- these store the integrand to be integrated over z
        oneHalo_ps_tz = np.zeros([nx,self.nZs])+0j
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_ps_tSZ = np.zeros([nx,self.nMasses])+0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.pk_profiles['y'][i,j]\
                                 *(1-np.exp(-(hcos.ks/hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                # Accumulate the integrands
                integrand_oneHalo_ps_tSZ[:,j] = y*np.conjugate(y)*hcos.nzm[i,j]

                # Perform the m integrals
            oneHalo_ps_tz[:,i]=np.trapz(integrand_oneHalo_ps_tSZ,hcos.ms,axis=-1)

        # Integrate over z
        ps_oneHalo_tSZ = self.T_CMB**2 * np.trapz( oneHalo_ps_tz * 1. / hcos.comoving_radial_distance(hcos.zs) ** 2\
                                                   * (hcos.h_of_z(hcos.zs)), hcos.zs, axis=-1)

        # ToDo: implement 2 halo term for tSZ PS tests
        ps_twoHalo_tSZ = np.zeros(ps_oneHalo_tSZ.shape)
        return ps_oneHalo_tSZ, ps_twoHalo_tSZ

    def get_cib_bias(self, exp, fftlog_way=True, bin_width_out=30):
        """
        Calculate the CIB biases given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
        """
        autofreq = np.array([[exp.freq_GHz], [exp.freq_GHz]], dtype=np.double)   *1e9    #Ghz
        hcos = self.hcos
        self.get_consistency(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the integrand to be integrated over z
        IIII_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIII_2h = IIII_1h.copy(); oneHalo_cross = IIII_1h.copy(); twoHalo_cross = IIII_1h.copy()

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            integrand_oneHalo_IIII=integrand_oneHalo_cross.copy(); integrand_twoHalo_k=integrand_oneHalo_cross.copy()
            integrand_twoHalo_II=integrand_oneHalo_cross.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                f_cen = tls.from_Jypersr_to_uK(exp.freq_GHz) * hcos._get_fcen(autofreq[0])[i,j][0]
                f_sat = tls.from_Jypersr_to_uK(exp.freq_GHz) * hcos._get_fsat(autofreq[0], cibinteg='trap', satmf='Tinker')[i,j][0]
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.uk_profiles['nfw'][i, j] * \
                                         (1-np.exp(-(hcos.ks/hcos.p['kstar_damping']))), ellmax=exp.lmax)

                phi_estimate_cfft_uu =  exp.get_TT_qe(fftlog_way, ells_out, u, u)

                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.uk_profiles['nfw'][i,j]\
                                   *hcos.lensing_window(hcos.zs[i],1100.), ellmax=self.lmax_out)
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]
                # Accumulate the integrands
                integrand_oneHalo_cross[...,j] = f_cen*(2*f_sat + f_sat**2) * phi_estimate_cfft_uu * np.conjugate(kfft) * hcos.nzm[i,j]
                integrand_oneHalo_IIII[...,j] = f_cen*(4*f_sat**3 + f_sat**4) * phi_estimate_cfft_uu * np.conjugate(phi_estimate_cfft_uu) * hcos.nzm[i,j]

                integrand_twoHalo_k[...,j] = np.conjugate(kfft) * hcos.nzm[i,j] * hcos.bh[i,j]
                integrand_twoHalo_II[...,j] = f_cen*(2*f_sat + f_sat**2) * phi_estimate_cfft_uu * hcos.nzm[i,j] * hcos.bh[i,j]

            # Perform the m integrals
            IIII_1h[...,i]=np.trapz(integrand_oneHalo_IIII,hcos.ms,axis=-1)

            oneHalo_cross[...,i]=np.trapz(integrand_oneHalo_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            IIII_2h[...,i] = np.trapz(integrand_twoHalo_II,hcos.ms,axis=-1)**2 * pk

            tmpCorr =np.trapz(integrand_twoHalo_k,hcos.ms,axis=-1)
            twoHalo_cross[...,i]=np.trapz(integrand_twoHalo_II,hcos.ms,axis=-1)\
                                 *(tmpCorr + hcos.lensing_window(hcos.zs,1100.)[i] - hcos.lensing_window(hcos.zs[i],1100.)\
                                   *self.consistency[i])*pk#

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # Integrand factors from Limber projection (adapted to hmvec conventions)
        IIII_integrand = (1+hcos.zs)**-4 * hcos.comoving_radial_distance(hcos.zs)**-6 * hcos.h_of_z(hcos.zs)**-1
        kII_integrand  = (1+hcos.zs)**-2 * hcos.comoving_radial_distance(hcos.zs)**-4

        # Integrate over z
        exp.biases['cib']['trispec']['1h'] = np.trapz( IIII_integrand*IIII_1h, hcos.zs, axis=-1)
        exp.biases['cib']['trispec']['2h'] = np.trapz( IIII_integrand*IIII_2h, hcos.zs, axis=-1)
        exp.biases['cib']['prim_bispec']['1h'] = conversion_factor * np.trapz( oneHalo_cross*kII_integrand,
                                                                               hcos.zs, axis=-1)
        exp.biases['cib']['prim_bispec']['2h'] = conversion_factor * np.trapz( twoHalo_cross*kII_integrand,
                                                                               hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['cib']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_cib_ps(self, exp):
        """
        Calculate the CIB power spectrum
        Input:
            * exp = a qest.experiment object
        """
        autofreq = np.array([[exp.freq_GHz], [exp.freq_GHz]], dtype=np.double)   *1e9    #Ghz
        hcos = self.hcos

        gal_prof_square = hcos._get_cib_square(autofreq, satflag=True, cibinteg='trap', satmf='Tinker')
        gal_prof = hcos._get_cib(autofreq[0], satflag=True, cibinteg='trap', satmf='Tinker')

        # Output ells
        ells_out = np.arange(self.lmax_out+1)

        nx = self.lmax_out+1

        # The one and two halo bias terms -- these store the integrand to be integrated over z
        oneHalo_ps = np.zeros([nx,self.nZs])+0j
        twoHalo_ps = np.zeros([nx,self.nZs])+0j
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_ps = np.zeros([nx,self.nMasses])+0j
            integrand_twoHalo_2g = np.zeros([nx, self.nMasses]) + 0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                g_square = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                    gal_prof_square[i, j] * (1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                    ellmax=self.lmax_out)
                g = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                             gal_prof[i, j] * (1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                # Accumulate the integrands
                integrand_oneHalo_ps[:, j] = g_square * hcos.nzm[i, j]
                integrand_twoHalo_2g[:, j] = g * hcos.nzm[i, j] * hcos.bh[i, j]

                # Perform the m integrals
            oneHalo_ps[:, i] = np.trapz(integrand_oneHalo_ps, hcos.ms, axis=-1)
            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=self.lmax_out)
            twoHalo_ps[:, i] = np.trapz(integrand_twoHalo_2g, hcos.ms, axis=-1) ** 2 * pk

        # Integrate over z
        clCIBCIB_oneHalo_ps = np.trapz(
            oneHalo_ps * (1 + hcos.zs) ** -2 * hcos.comoving_radial_distance(hcos.zs) ** -2 * (hcos.h_of_z(hcos.zs) ** -1),
            hcos.zs, axis=-1)
        clCIBCIB_twoHalo_ps = np.trapz(
            twoHalo_ps * (1 + hcos.zs) ** -2 * hcos.comoving_radial_distance(hcos.zs) ** -2 * (hcos.h_of_z(hcos.zs) ** -1),
            hcos.zs, axis=-1)
        return clCIBCIB_oneHalo_ps, clCIBCIB_twoHalo_ps

    def get_mixed_biases(self, exp, fftlog_way=True, bin_width_out=30):
        """
        Calculate the biases involving both CIB and tSZ given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
        """
        autofreq = np.array([[exp.freq_GHz], [exp.freq_GHz]], dtype=np.double)   *1e9    #Ghz
        hcos = self.hcos
        self.get_consistency(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the integrand to be integrated over z
        Iyyy_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIyy_1h = Iyyy_1h.copy(); yIII_1h = Iyyy_1h.copy(); Iyyy_2h = Iyyy_1h.copy(); IIyy_2h = Iyyy_1h.copy()
        yIII_2h = Iyyy_1h.copy(); oneHalo_cross = Iyyy_1h.copy(); twoHalo_cross = Iyyy_1h.copy()

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            integrand_oneHalo_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            integrand_oneHalo_Iyyy=integrand_oneHalo_cross.copy(); integrand_oneHalo_IIyy=integrand_oneHalo_cross.copy()
            integrand_oneHalo_yIII=integrand_oneHalo_cross.copy(); integrand_twoHalo_k=integrand_oneHalo_cross.copy()
            integrand_twoHalo_yy=integrand_oneHalo_cross.copy(); integrand_twoHalo_Iy=integrand_oneHalo_cross.copy()
            integrand_twoHalo_II=integrand_oneHalo_cross.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                f_cen = tls.from_Jypersr_to_uK(exp.freq_GHz) * hcos._get_fcen(autofreq[0])[i,j][0]
                f_sat = tls.from_Jypersr_to_uK(exp.freq_GHz) * hcos._get_fsat(autofreq[0], cibinteg='trap', satmf='Tinker')[i,j][0]
                #project the galaxy profiles
                y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.pk_profiles['y'][i, j] \
                                 * (1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.uk_profiles['nfw'][i, j] * \
                                         (1-np.exp(-(hcos.ks/hcos.p['kstar_damping']))), ellmax=exp.lmax)

                phi_estimate_cfft_uu =  exp.get_TT_qe(fftlog_way, ells_out, u, u)
                phi_estimate_cfft_uy =  exp.get_TT_qe(fftlog_way, ells_out, u, y)
                phi_estimate_cfft_yy = exp.get_TT_qe(fftlog_way, ells_out, y, y)

                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.uk_profiles['nfw'][i,j]\
                                   *hcos.lensing_window(hcos.zs[i],1100.), ellmax=self.lmax_out)
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]
                # Accumulate the integrands
                integrand_oneHalo_cross[...,j] = (f_cen + f_sat) * phi_estimate_cfft_uy * np.conjugate(kfft) * hcos.nzm[i,j]
                integrand_oneHalo_Iyyy[...,j] = (f_cen + f_sat) * phi_estimate_cfft_uy * phi_estimate_cfft_yy * hcos.nzm[i,j]
                integrand_oneHalo_IIyy[...,j] = f_cen*(2*f_sat + f_sat**2) * phi_estimate_cfft_uu * phi_estimate_cfft_yy * hcos.nzm[i,j]
                integrand_oneHalo_yIII[...,j] = f_cen*(3*f_sat**2 + f_sat**3) * phi_estimate_cfft_uu * phi_estimate_cfft_uy * hcos.nzm[i,j]

                integrand_twoHalo_k[...,j] = np.conjugate(kfft) * hcos.nzm[i,j] * hcos.bh[i,j]
                integrand_twoHalo_yy[...,j] = phi_estimate_cfft_yy * hcos.nzm[i,j] * hcos.bh[i,j]
                integrand_twoHalo_Iy[...,j] = (f_cen + f_sat) * phi_estimate_cfft_uy * hcos.nzm[i,j] * hcos.bh[i,j]
                integrand_twoHalo_II[...,j] = f_cen*(2*f_sat + f_sat**2) * phi_estimate_cfft_uu * hcos.nzm[i,j] * hcos.bh[i,j]

            # Perform the m integrals
            Iyyy_1h[...,i]=np.trapz(integrand_oneHalo_Iyyy,hcos.ms,axis=-1)
            IIyy_1h[...,i]=np.trapz(integrand_oneHalo_IIyy,hcos.ms,axis=-1)
            yIII_1h[...,i]=np.trapz(integrand_oneHalo_yIII,hcos.ms,axis=-1)

            oneHalo_cross[...,i]=np.trapz(integrand_oneHalo_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            Iyyy_2h[...,i] = np.trapz(integrand_twoHalo_Iy,hcos.ms,axis=-1) * np.trapz(integrand_twoHalo_yy,hcos.ms,axis=-1) * pk
            IIyy_2h[...,i] = np.trapz(integrand_twoHalo_II,hcos.ms,axis=-1) * np.trapz(integrand_twoHalo_yy,hcos.ms,axis=-1) * pk
            yIII_2h[...,i] = np.trapz(integrand_twoHalo_Iy,hcos.ms,axis=-1) * np.trapz(integrand_twoHalo_II,hcos.ms,axis=-1) * pk

            tmpCorr =np.trapz(integrand_twoHalo_k,hcos.ms,axis=-1)
            twoHalo_cross[...,i]=np.trapz(integrand_twoHalo_Iy,hcos.ms,axis=-1)\
                                 *(tmpCorr + hcos.lensing_window(hcos.zs,1100.)[i] - hcos.lensing_window(hcos.zs[i],1100.)\
                                   *self.consistency[i])*pk#

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # Integrand factors from Limber projection (adapted to hmvec conventions)
        Iyyy_integrand = 4 * (1+hcos.zs)**-1 * hcos.comoving_radial_distance(hcos.zs)**-6 * hcos.h_of_z(hcos.zs)**2
        IIyy_integrand = 2 * (1+hcos.zs)**-2 * hcos.comoving_radial_distance(hcos.zs)**-6 * hcos.h_of_z(hcos.zs)
        yIII_integrand = 4 * (1+hcos.zs)**-3 * hcos.comoving_radial_distance(hcos.zs)**-6
        kIy_integrand  = 2 * (1+hcos.zs)**-1 * hcos.comoving_radial_distance(hcos.zs)**-4 * hcos.h_of_z(hcos.zs)

        # Integrate over z
        exp.biases['mixed']['trispec']['1h'] = np.trapz( Iyyy_integrand*Iyyy_1h + IIyy_integrand*IIyy_1h
                                                       + yIII_integrand*yIII_1h, hcos.zs, axis=-1)
        exp.biases['mixed']['trispec']['2h'] = np.trapz( Iyyy_integrand*Iyyy_2h + IIyy_integrand*IIyy_2h
                                                       + yIII_integrand*yIII_2h, hcos.zs, axis=-1)
        exp.biases['mixed']['prim_bispec']['1h'] = conversion_factor * np.trapz( oneHalo_cross*kIy_integrand,
                                                                               hcos.zs, axis=-1)
        exp.biases['mixed']['prim_bispec']['2h'] = conversion_factor * np.trapz( twoHalo_cross*kIy_integrand,
                                                                               hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['mixed']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return
