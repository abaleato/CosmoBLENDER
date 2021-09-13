# Temporary file storing the functions needed to evaluate thesecondary bispectrum bias. The idea is to use this as a
# test bed, and then move all this functionality into other files which already exist.

import numpy as np
import quicklens as ql
from lensing_rec_biases_code import qest
import multiprocessing
from functools import partial

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

def get_secondary_bispec_bias_at_L(projected_y_profile, projected_kappa_profile, exp_param_list, L):
    # FIXME: careful with the 'real' assertions here
    # FIXME: Dont you need some factors to go from discrete to cts?
    # FIXME: Various prefactors (mostly the three factors of 2pi when following Lewis & Challinor conventions)
    # TODO: Add 2h term
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

    # Wiener filters in T. Depends on |\vec{l}''|. Note the use of lensed cltt to optimise lensing reconstruction
    W_T = ql.spec.cl2cfft(experiment.cl_len.cltt / (experiment.cl_len.cltt + experiment.nltt), experiment.pix).fft
    # The factors that depend on |\vec{L} - \vec{l}''|. Assume \vec{L} points along x axis.
    T_fg_filtered_shifted = shift_array(projected_y_profile / (experiment.cl_len.cltt + experiment.nltt)
                                  , experiment, L).fft.real

    # Calculate the inner reconstruction
    inner_rec = get_inner_reconstruction(experiment, T_fg_filtered_shifted, projected_kappa_profile).fft

    # Carry out the 2D integral over x and y
    # Note that the \vec{L} \cdot \vec{l} simplifies because we assume that L is aligned with x-axis
    # We roll the arrays so that that domain of integration is sorted sequentially as required by np.trapz

    full_integrand = np.roll( np.roll(lx * W_T * inner_rec * T_fg_filtered_shifted, experiment.nx // 2, axis=-1),
                              experiment.nx // 2, axis = 0)

    integral_over_x = np.trapz(full_integrand, np.roll(lx[0, :], experiment.nx//2), axis=-1)

    # TODO: implement experiment.ny in addition to experiment.nx
    integral_over_y = np.trapz( integral_over_x, np.roll(ly[:, 0], experiment.nx//2, axis=0), axis=-1)

    # Normalise the two QEs involved here
    lbins = np.arange(0, experiment.lmax, 40)
    qe_norm_1D = experiment.qe_norm.get_ml(lbins)
    qe_norm_at_L = np.interp(L, qe_norm_1D.ls, qe_norm_1D.specs['cl'])

    return (-2) / np.pi / (2*np.pi)**2 * L**2 * integral_over_y / (qe_norm_at_L ** 2)

def shift_array(array_to_paste, exp, lx_shift, ly_shift=0):
    """
    - Inputs:
        * array_to_paste = 1D numpy array. The 1D y/density profile that we want to paste in 2D and shift.
        * exp = qest.exp object. The experiment from which to get nx, dx, etc.
        * lx_shift = Float or int. Multipole by which to shift the centre of the array in the x direction.
        * ly_shift = Float or int. Multipole by which to shift the centre of the array in the y direction.
    - Returns:
        * A ql.maps.cfft object containing the 2D, shifted array_to_paste.
    """
    lx, ly = ql.maps.cfft(exp.nx, exp.dx).get_lxly()
    shifted_ells = np.sqrt((lx_shift + lx) ** 2 + (ly_shift + ly) ** 2).flatten()
    return ql.maps.cfft(exp.nx, exp.dx, fft=np.interp(shifted_ells, exp.ls, array_to_paste).reshape(exp.nx, exp.nx) )

def get_inner_reconstruction(experiment, T_fg_filtered_shifted, projected_kappa_profile):
    """ Evaluate the (unnormalised) inner QE reconstructions in the calculation of the secondary bispectrum bias.
        We assume a fixed L so that we can use the convolution theorem.
    """
    # TODO: Document better
    # TODO: check prefactors of 2pi
    # TODO: write some tests

    ret = ql.maps.cfft(nx=experiment.pix.nx, dx=experiment.pix.dx, ny=experiment.pix.ny, dy=experiment.pix.dy)

    lx, ly = ret.get_lxly()

    # The cl factors that depend on |\vec{l}''|
    cltt_filters = ql.spec.cl2cfft(experiment.cl_unl.cltt * experiment.cl_len.cltt /
                                   (experiment.cl_len.cltt + experiment.nltt), experiment.pix).fft

    # Convert kappa to phi and paste onto grid
    phi = ql.spec.cl2cfft(np.nan_to_num(projected_kappa_profile / (0.5 * experiment.cl_unl.ls *
                                        (experiment.cl_unl.ls + 1))), experiment.pix).fft

    # Carry out integral using convolution theorem
    ret.fft = np.fft.fft2(np.fft.ifft2(cltt_filters * T_fg_filtered_shifted * lx**2 ) * np.fft.ifft2(lx * phi)) \
              +  np.fft.fft2(np.fft.ifft2(cltt_filters * T_fg_filtered_shifted * lx * ly) * np.fft.ifft2(ly * phi))

    # Correct for numpy DFT normalisation correction and go from discrete to continuous
    # FIXME: make sure these factors are correct
    ret *= 1 / (ret.dx * ret.dy)

    return ret
