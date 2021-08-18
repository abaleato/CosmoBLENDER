# Temporary file storing the functions needed to evaluate thesecondary bispectrum bias. The idea is to use this as a
# test bed, and then move all this functionality into other files which already exist.

import numpy as np
import quicklens as ql
from lensing_rec_biases_code import qest
import multiprocessing
from functools import partial

def get_secondary_bispec_bias_at_L(projected_y_profile, projected_kappa_profile, exp_param_list, L):
    # FIXME: careful with the 'real' assertions here
    # FIXME: Dont you need some factors to go from discrete to cts?
    # FIXME: Various prefactors (mostly the three factors of 2pi when following Lewis & Challinor conventions)
    """ Calculate the outer QE reconstruction in the secondary bispectrum bias, for given profiles.
        This involves brute-force integration of a shifted tsz profile times another, "inner" reconstruction,
        which we call along the way. We align L with the x-axis.
    - Inputs:
        * L = Float or int. Multipole at which to return the bias
        * projected_y_profile = 1D numpy array. Project y/density profile.
        * projected_kappa_profile = 1D numpy array. The 1D projected kappa that we will paste to 2D.
        * exp_param_list = list of inputs to initialise the 'exp' experiment object

    - Returns:
        * A Float. The value of the secondary bispectrum bias at L.
    """
    # Initialise experiment object. (We do this bc instances cannot be passed via the multiprocessing pipe)
    experiment = qest.experiment(*exp_param_list)

    lx, ly = ql.maps.cfft(experiment.nx, experiment.dx).get_lxly()
    lmag = np.sqrt(lx ** 2 + ly ** 2)
    lmag[0] = 1.  # To avoid nans # FIXME: check that this does not affect the calculation
    # One of the legs of the QE, with Wiener filtering
    shifted_y_array = shift_array(projected_y_profile / (experiment.cl_len.cltt + experiment.nltt) \
                                  , experiment, L)

    # Calculate the inner reconstruction using QL, featuring the appropriate ells from the lensing weights
    # (Note the division by lmag**2 so we can input kappa instead of phi, the factor of -2 is put in later on)
    # Note that QL will automatically put in two factors of Wiener filters (out of the total of 4 we need)
    inner_rec = get_inner_rec(L, experiment, shifted_y_array, projected_kappa_profile, lx, lx / (lmag ** 2)) + \
                get_inner_rec(L, experiment, shifted_y_array, projected_kappa_profile, ly, ly / (lmag ** 2)) #FIXME: can this be further simplified?

    # Carry out the 2D integral over x and y
    # We include a factor of the Wiener filters and the Cls coming coming the lensing weights
    # Note that the \vec{L} \cdot \vec{l} simplifies because we assume that L is aligned with x-axis
    # FIXME: only implementing one of two terms in g(l,L); is this OK?
    integral_over_x_w_l = np.trapz(
        lx * ql.spec.cl2cfft(experiment.cl_unl.cltt / (experiment.cl_len.cltt + experiment.nltt), experiment.pix).fft \
        * inner_rec * shifted_y_array.fft.real, lx[0, :], axis=-1)
    integral_over_x = np.trapz(
        ql.spec.cl2cfft(experiment.cl_unl.cltt / (experiment.cl_len.cltt + experiment.nltt), experiment.pix).fft \
        * inner_rec * shifted_y_array.fft.real, lx[0, :], axis=-1)

    integral_over_y = L**2 * np.trapz(integral_over_x, ly[:, 0], axis=-1) \
                      - L * np.trapz(integral_over_x_w_l, ly[:, 0], axis=-1)

    # Normalise the two QEs involved here
    lbins = np.arange(0, experiment.lmax, 40)
    qe_norm_1D = experiment.qe_norm.get_ml(lbins)
    qe_norm_at_L = np.interp(L, qe_norm_1D.ls, qe_norm_1D.specs['cl'])

    # Prefactor of 8 from permutations, (-2) for going from phi to kappa ,
    # (2pi)^-1 from 1st order expansion in lensing, (-1) that's in the analytic calculation
    # and (2pi)^-1 from QE (another should be built into QL implementation)
    return (-1) * 8 * (-2) * (2 * np.pi) ** (-2) * integral_over_y / qe_norm_at_L ** 2

def get_inner_rec(L, experiment, projected_y_profile, projected_kappa_profile, aux_array_leg1=1, aux_array_leg2=1,
                  key='ptt'):
    """ Inner QE reconstruction in the evaluation of the secondary bispectrum bias
    - Inputs:
        * L = Float or int. The multipole at which we're evaluating the bias. We assume it's assigned with x-axis.
        * experiment = qest.exp object. The experiment from which to get nx, dx, etc.
        * projected_y_profile = A ql.maps.cfft object containing the 2D, shifted y profile.
        * projected_kappa_profile = 1D numpy array. The 1D projected kappa that we will paste to 2D.
        * aux_array_leg1 = 2D numpy array. Ell weighting to apply to leg 1 of the QE
        * aux_array_leg2 = 2D numpy array. Ell weighting to apply to leg 2 of the QE
        * key = String. Quadratic estimator key to use (in QL nomenclature). Unlikely to ever use anything but 'ptt'
    - Returns:
        * A ql.maps.cfft object containing the 2D, shifted array_to_paste.
    """
    tft1 = ql.spec.cl2cfft(experiment.cl_unl.cltt, experiment.pix) * projected_y_profile
    tft2 = ql.spec.cl2cfft(projected_kappa_profile, experiment.pix)

    # Apply filters and weights prior to lensing reconstructions
    t_filter = experiment.ivf_lib.get_fl().get_cffts()[0]
    tft1.fft *= aux_array_leg1 * t_filter.fft
    tft2.fft *= aux_array_leg2 * t_filter.fft

    unnormalized_phi = experiment.qest_lib.get_qft(key, tft1, 0 * tft1.copy(), 0 * tft1.copy(), tft2, 0 * tft1.copy(),
                                                   0 * tft1.copy())

    # In QL, the unnormalised reconstruction (obtained via eval_flatsky()) comes with a factor of sqrt(skyarea)
    A_sky = (experiment.dx * experiment.nx) ** 2
    # Normalize the reconstruction
    return np.nan_to_num(unnormalized_phi.fft[:, :]) / np.sqrt(A_sky)

def shift_array(array_to_paste, exp, lx_shift, ly_shift=0):
    """
    - Inputs:
        * array_to_paste = 1D numpy array. The 1D y/density profile that we want to paste in 2D and shift.
        * exp = qest.exp object. The experiment from which to get nx, dx, etc.
        * lx_shift = Float or int. Multipole by which to shift the centre fo the array in the x direction.
        * ly_shift = Float or int. Multipole by which to shift the centre fo the array in the y direction.
    - Returns:
        * A ql.maps.cfft object containing the 2D, shifted array_to_paste.
    """
    lx, ly = ql.maps.cfft(exp.nx, exp.dx).get_lxly()
    shifted_ells = np.sqrt((lx_shift + lx) ** 2 + (ly_shift + ly) ** 2).flatten()
    return ql.maps.cfft(exp.nx, exp.dx, fft=np.interp(shifted_ells, exp.ls, \
                                                      array_to_paste).reshape(exp.nx, exp.nx))

def get_secondary_bispec_bias(lbins, exp_param_list, projected_y_profile, projected_kappa_profile, parallelise=True):
    """
    Calculate contributions to secondary bispectrum bias from given profiles, either serially or via multiple processes
    Input:
        * lbins = np array. bins centres at which to evaluate the secondary bispec bias
        * exp_param_list = list of inputs to initialise the 'exp' experiment object
        * projected_y_profile = 1D numpy array. Project y/density profile.
        * projected_kappa_profile = 1D numpy array. The 1D projected kappa that we will paste to 2D.
        parallelise = Boolean. If True, use multiple processes. Alternatively, proceed serially.
    - Returns:
        * A np array with the size of lbins containing contributions to the secondary bispectrum bias
    """

    if parallelise:
        # Use multiprocessing to speed up calculation
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) # Start as many processes as machine can handle
        # Helper function (pool.map can only take one, iterable input)
        func = partial(get_secondary_bispec_bias_at_L, projected_y_profile, projected_kappa_profile, exp_param_list)
        second_bispec_bias = np.array(pool.map(func, lbins))
        pool.close()

    else:
        second_bispec_bias = np.zeros(lbins.shape)
        for i, L in enumerate(lbins):
            second_bispec_bias[i] = get_secondary_bispec_bias_at_L(projected_y_profile, projected_kappa_profile, exp_param_list, L)
    return second_bispec_bias
