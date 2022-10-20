"""

ariel@oapd
30/09/2022


"""

import numpy as np


def calc_manual_ew(wl, flux, delta_wl, line):

    if line == 'Hd':

        blue_cont = np.median(flux[(wl > 4076.4) & (wl < 4088.8)])
        red_cont = np.median(flux[(wl > 4117.2) & (wl < 4136.7)])

        blue_wl = np.mean(wl[(wl > 4076.4) & (wl < 4088.8)])
        red_wl = np.mean(wl[(wl > 4117.2) & (wl < 4136.7)])

        ew_range = (wl > 4091.75) & (wl < 4112)

    if line == 'Hb':

        blue_cont = np.median(flux[(wl > 4806.0) & (wl < 4826.0)])
        red_cont = np.median(flux[(wl > 4896.0) & (wl < 4918.0)])

        blue_wl = np.mean(wl[(wl > 4806.0) & (wl < 4826.0)])
        red_wl = np.mean(wl[(wl > 4896.0) & (wl < 4918.0)])

        ew_range = (wl > 4826.0) & (wl < 4896.0)

    if line == 'Ha':

        blue_cont = np.median(flux[(wl > 6505.0) & (wl < 6535.0)])
        red_cont = np.median(flux[(wl > 6595.0) & (wl < 6625.0)])

        blue_wl = np.mean(wl[(wl > 6505.0) & (wl < 6535.0)])
        red_wl = np.mean(wl[(wl > 6595.0) & (wl < 6625.0)])

        ew_range = (wl > 6553.0) & (wl < 6573.0)

    if line == 'Ha':

        blue_cont = np.median(flux[(wl > 6505.0) & (wl < 6535.0)])
        red_cont = np.median(flux[(wl > 6595.0) & (wl < 6625.0)])

        blue_wl = np.mean(wl[(wl > 6505.0) & (wl < 6535.0)])
        red_wl = np.mean(wl[(wl > 6595.0) & (wl < 6625.0)])

        ew_range = (wl > 6553.0) & (wl < 6573.0)

    cont_slope = (red_cont - blue_cont) / (red_wl - blue_wl)
    cont_intercept = blue_cont - blue_wl * cont_slope

    cont_level = wl * cont_slope + cont_intercept

    equivalent_width = np.trapz(1 - (flux[ew_range] / cont_level[ew_range]), dx=delta_wl)

    return equivalent_width
