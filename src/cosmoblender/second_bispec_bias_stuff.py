# Temporary file storing the functions needed to evaluate thesecondary bispectrum bias. The idea is to use this as a
# test bed, and then move all this functionality into other files which already exist.

import numpy as np
import quicklens as ql
from . import qest
import multiprocessing
from functools import partial

def get_sec_bispec_bias(L_array, qe_norm_1D, exp_param_dict, cltt_tot, projected_fg_profile_1, \
                              projected_kappa_profile, projected_fg_profile_2=None, parallelise=False):
    """
    Calculate contributions to secondary bispectrum bias from given profiles, either serially or via multiple processes
    Input:
        * L_array = np array. bins centres at which to evaluate the secondary bispec bias
        * qe_norm_1D = 1D arraycontaining normalisation of the QE at L_array multipoles
        * exp_param_dict = dict of kwargs to initialise a bare-bones 'exp' experiment object with
        * cltt_tot = 1D np array. Total TT power, possibly after fg cleaning
        * projected_fg_profile_1 = 1D numpy array. Project y/density profile.
        * projected_kappa_profile = 1D numpy array. The 1D projected kappa that we will paste to 2D.
        * projected_fg_profile_2 (optional) = 1D numpy array. Project y/density profile. By default, equals projected_fg_profile_1
        * parallelise = Boolean. If True, use multiple processes. Alternatively, proceed serially.

    - Returns:
        * A np array with the size of L_array containing contributions to the secondary bispectrum bias
    """
    if projected_fg_profile_2 is None:
        projected_fg_profile_2 = projected_fg_profile_1

    # Precompute some useful quantities
    experiment = qest.experiment(**exp_param_dict, bare_bones=True)
    W_T = ql.spec.cl2cfft(experiment.cl_len.cltt / cltt_tot, experiment.pix).fft.real
    cltt_filters = ql.spec.cl2cfft(experiment.cl_unl.cltt * experiment.cl_len.cltt / cltt_tot,
                                   experiment.pix).fft.real
    # Convert kappa to phi and paste onto grid
    phi_gridded = ql.spec.cl2cfft(np.nan_to_num(projected_kappa_profile / (0.5 * experiment.cl_unl.ls *
                                        (experiment.cl_unl.ls + 1))), experiment.pix).fft.real
    ret = ql.maps.cfft(nx=experiment.pix.nx, dx=experiment.pix.dx, ny=experiment.pix.ny, dy=experiment.pix.dy)
    lx, ly = ret.get_lxly()
    lxly_tuple = [lx, ly]
    l_phi_ifft_tuple = [np.fft.ifft2(lx * phi_gridded), np.fft.ifft2(ly * phi_gridded)]

    if parallelise:
        # TODO: there's probably better ways than to count explicitly the number of cores
        # Use multiprocessing to speed up calculation
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1) # Start as many processes as machine can handle
        # Helper function (pool.map can only take one, iterable input)
        func = partial(get_sec_bispec_bias_at_L, projected_fg_profile_1, projected_fg_profile_2, experiment.pix,
                       cltt_tot, W_T, cltt_filters, phi_gridded, lxly_tuple, l_phi_ifft_tuple)
        second_bispec_bias = np.array(pool.map(func, L_array))
        pool.close()

    else:
        second_bispec_bias = np.zeros(L_array.shape)
        for i, L in enumerate(L_array):
            print(L)
            second_bispec_bias[i] = get_sec_bispec_bias_at_L(projected_fg_profile_1, projected_fg_profile_2,
                                                             experiment.pix, cltt_tot, W_T, cltt_filters, phi_gridded,
                                                             lxly_tuple, l_phi_ifft_tuple, L)
    # Finally, normalise the reconstruction
    return second_bispec_bias / (qe_norm_1D ** 2)

def get_sec_bispec_bias_at_L(projected_fg_profile_1, projected_fg_profile_2, pix, cltt_tot, W_T,
                             cltt_filters, phi_gridded, lxly_tuple, l_phi_ifft_tuple, L):
    # TODO: Add 2h term
    # TODO: document new args
    """ Calculate the (unnormalized) outer QE reconstruction in the secondary bispectrum bias, for given profiles.
        This involves brute-force integration of a shifted tsz profile times another, "inner" reconstruction,
        which we call along the way. We align L with the x-axis.
    - Inputs:
        * projected_fg_profile_1 = 1D numpy array. Project y/density profile.
        * projected_fg_profile_2 = 1D numpy array. Project y/density profile.
        * projected_kappa_profile = 1D numpy array. The 1D projected kappa that we will paste to 2D.
        * pix = a ql.maps.pix object with information of the pixelization
        * cltt_tot = 1D numpy array. Total TT power including CMB, noise and fgs, possibly after fg cleaning
        * L = Float or int. Multipole at which to return the bias
    - Returns:
        * A Float. The value of the secondary bispectrum bias at L.
    """
    lx, ly = lxly_tuple

    # The factors that depend on |\vec{L} - \vec{l}''|. Assume \vec{L} points along x axis.
    T_fg_filtered_shifted_1 = shift_array(projected_fg_profile_1 / cltt_tot, pix, lxly_tuple, L).fft.real
    T_fg_filtered_shifted_2 = shift_array(projected_fg_profile_2 / cltt_tot, pix, lxly_tuple, L).fft.real

    # Calculate the inner reconstruction
    inner_rec = get_inner_reconstruction(pix, T_fg_filtered_shifted_1, phi_gridded, cltt_filters, l_phi_ifft_tuple).fft.real

    # Carry out the 2D integral over x and y
    # Note that the \vec{L} \cdot \vec{l} simplifies because we assume that L is aligned with x-axis
    # We roll the arrays so that that domain of integration is sorted sequentially as required by np.trapz
    full_integrand = np.roll( np.roll(lx * W_T * inner_rec * T_fg_filtered_shifted_2, pix.nx // 2, axis=-1),
                              pix.nx // 2, axis = 0)

    integral_over_x = np.trapz(full_integrand, np.roll(lx[0, :], pix.nx//2, axis=-1), axis=-1)

    # TODO: implement experiment.ny in addition to experiment.nx
    integral_over_y = np.trapz( integral_over_x, np.roll(ly[:, 0], pix.nx//2, axis=0), axis=-1)
    return (-2) / (2*np.pi)**3 * L**2 * integral_over_y

def shift_array(array_to_paste, pix, lxly_tuple, lx_shift, ly_shift=0):
    """
    - Inputs:
        * array_to_paste = 1D numpy array. The 1D y/density profile that we want to paste in 2D and shift.
        * exp = qest.exp object. The experiment from which to get nx, dx, etc.
        * lx_shift = Float or int. Multipole by which to shift the centre of the array in the x direction.
        * ly_shift = Float or int. Multipole by which to shift the centre of the array in the y direction.
    - Returns:
        * A ql.maps.cfft object containing the 2D, shifted array_to_paste.
    """
    lx, ly = lxly_tuple
    shifted_ells = np.sqrt((lx_shift + lx) ** 2 + (ly_shift + ly) ** 2).flatten()
    return ql.maps.cfft(pix.nx, pix.dx, fft=np.interp(shifted_ells, np.arange(len(array_to_paste)), array_to_paste).reshape(pix.nx, pix.nx) )

def get_inner_reconstruction(pix, T_fg_filtered_shifted, phi_gridded, cltt_filters, l_phi_ifft_tuple):
    """ Evaluate the (unnormalised) inner QE reconstructions in the calculation of the secondary bispectrum bias.
        We assume a fixed L so that we can use the convolution theorem.
    """
    # TODO: Document better
    # TODO: check prefactors of 2pi

    ret = ql.maps.cfft(nx=pix.nx, dx=pix.dx, ny=pix.ny, dy=pix.dy)
    lx, ly = ret.get_lxly()

    lx_phi_ifft, ly_phi_ifft = l_phi_ifft_tuple

    # Carry out integral using convolution theorem
    ret.fft = np.fft.fft2(np.fft.ifft2(cltt_filters * T_fg_filtered_shifted * lx**2 ) * lx_phi_ifft) \
              +  np.fft.fft2(np.fft.ifft2(cltt_filters * T_fg_filtered_shifted * lx * ly) * ly_phi_ifft)

    # Correct for numpy DFT normalisation correction and go from discrete to continuous
    # FIXME: make sure these factors are correct
    ret *= 1 / (ret.dx * ret.dy)
    return ret

def test_integral_eval_at_L(exp_param_dict, L):
    ''' Test the evaluation of integrals by calculating the QE normalisation computed by brute force
        with the result given by Quicklens'''
    # TODO: delete this function once secondary bispec calculation works

    # Initialise a bare-bones experiment object. (We do this bc instances cannot be passed via the multiprocessing pipe)
    experiment = qest.experiment(**exp_param_dict, bare_bones=True)

    lx, ly = ql.maps.cfft(experiment.nx, experiment.dx).get_lxly()

    # Wiener filters in T. Depends on |\vec{l}''|. Note the use of lensed cltt to optimise lensing reconstruction
    W_T = ql.spec.cl2cfft(experiment.cl_len.cltt / experiment.cltt_tot, experiment.pix).fft

    # Carry out the 2D integral over x and y
    # Note that the \vec{L} \cdot \vec{l} simplifies because we assume that L is aligned with x-axis
    # We roll the arrays so that that domain of integration is sorted sequentially as required by np.trapz
    full_integrand = np.roll(np.roll(lx * (lx - L) * W_T , experiment.nx // 2, axis=-1),
                             experiment.nx // 2, axis=0)

    integral_over_x = np.trapz(full_integrand, np.roll(lx[0, :], experiment.nx // 2, axis=-1), axis=-1)
    integral_over_y = np.trapz(integral_over_x, np.roll(ly[:, 0], experiment.nx // 2, axis=0), axis=-1)
    return L**2 * integral_over_y  / (2*np.pi)**2
