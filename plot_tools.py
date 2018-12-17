import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astroML.plotting import scatter_contour
import starlight_toolkit.plotting as stplot
import starlight_toolkit.output as stout
import wololo

#Add one more color to seaborn pallete:
colors = sns.color_palette()
colors.append((0.86, 0.33999999999999997, 0.69879999999999964))

band_colors = {'FUV' : colors[5], 'NUV' : colors[0], 'u' : colors[3]
, 'g' : colors[1], 'r' : colors[2], 'i' : colors[6], 'z' : colors[4] }


def get_pred_interval(x,y,x_fit,y_fit,y_pred,n):
    '''
    Gets 2 sigma prediction bands for a fit with n degrees of freedom
    '''

    #Setting t value
    t = 3.182

    #Calculating prediction bands:
    sigma_y = np.sum((y - y_pred)**2 )/(n-2)
    sigma_x = np.sum( (x - np.mean(x))**2 )
    lim = []
    for i in range(len(x_fit)):
        lim = np.append(lim
        , t * np.sqrt(sigma_y) * np.sqrt( 1 + (1/n) + (((x_fit[i] - np.mean(x))**2)/(sigma_x ))))
    upper_pred = y_fit + lim
    lower_pred = y_fit - lim

    return [lower_pred, upper_pred]


def get_percentilebars(percentiles, data):

    percbar = []

    print(percbar)
    for a in range(len(percentiles)):
        percbar.append(( [np.absolute(percentiles[a][0]) - np.absolute(data[a])]
        , [np.absolute(data[a]) - np.absolute(percentiles[a][1])] ))
    return percbar


def errorplot(xdata, ydata, errors, datacolor):
    for i in range(len(xdata)):
        plt.errorbar(xdata[i], ydata[i], yerr = errors[i], fmt='o', color =
        datacolor, capsize = 10, zorder = 30)
        print('errors =', errors[i])


def hist2dscatter(x, y, nbins, threshold_value, axis, ms=1):
    scatter_contour(x, y, threshold=threshold_value, log_counts=True, ax=axis,
    histogram2d_args=dict(bins=nbins),
    plot_args=dict(marker='.', markersize = ms, linestyle='none'
    , color=plt.cm.plasma.colors[0]),
    contour_args=dict(cmap=plt.cm.plasma))


def bin_data(binvar, nbins=10, hist_range=None):
    if hist_range==None:
        hist_range = (np.percentile(binvar, 0.5),np.percentile(binvar, 99.5))
    hist, bin_edges = np.histogram(binvar, range=hist_range , bins=nbins)
    flagarray = np.array([(binvar > bin_edges[i]) & (binvar < bin_edges[i+1]) for i in range(len(bin_edges)-1)])
    return flagarray, bin_edges


def plot_average_in_bins(x, y, label='', color='k', nbins=10, ax=None):

    x_flag, x_bins = bin_data(x, nbins)

    y_means  = np.array([np.mean(y[x_flag[i]]) for i in range(nbins)])
    x_values = np.array([(x_bins[i]+x_bins[i+1])/2 for i in range(nbins)])

    if ax==None:
        ax=plt.gca()

    ax.plot(x_values, y_means, color=color, label=label)


def plot_median_in_bins(x, y, label='', color='g', nbins=10, ax=None, plot_percentiles=True, percentiles=[25,75],
                        percentiles_alpha=0.4, percentiles_color='g', plot_points=True, median_lw=2):

    x_flag, x_bins = bin_data(x, nbins)

    if type(nbins) == list:
        y_med    = np.array([np.median(y[x_flag[i]]) for i in range(len(nbins)-1)])
        x_values = np.array([(x_bins[i]+x_bins[i+1])/2 for i in range(len(nbins)-1)])
    else:
        y_med    = np.array([np.median(y[x_flag[i]]) for i in range(nbins)])
        x_values = np.array([(x_bins[i]+x_bins[i+1])/2 for i in range(nbins)])

    if ax==None:
        ax=plt.gca()

    ax.plot(x_values, y_med, color=color, label=label, lw=median_lw)

    if plot_points==True:
        ax.scatter(x_values, y_med, color=color, edgecolor='k', zorder=10)

    if plot_percentiles==True:
        y_low    = np.array([np.percentile(y[x_flag[i]], percentiles[0]) for i in range(nbins)])
        y_upp    = np.array([np.percentile(y[x_flag[i]], percentiles[1]) for i in range(nbins)])
        ax.fill_between(x_values, y_low, y_upp, alpha=percentiles_alpha, color=percentiles_color)

    return x_flag, x_bins



def plot_contours(x, y, contour_colors=None, contour_bins=None, hist_range=None, contour_levels=None, ax=None):
    if ax == None:
        ax = plt.gca()
    if contour_bins == None:
        contour_bins = 100

    if hist_range == None:
        hist_range = [[np.percentile(x,5),np.percentile(x,95)], [np.percentile(y,5),np.percentile(y,95)]]

    H, xedges, yedges = np.histogram2d(x,y, range=hist_range, bins=contour_bins, normed=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.contour(H.transpose(), color=contour_colors, linewidths=2, extent=extent, levels=contour_levels)



def plot_starlight_comparison(file_PHO, file_OPT, label1, label2, ax=None):
    #Create figure with axis:

    if ax==None:
        axis = plt.gca()
    else:
        axis=ax

    out_OPT = stout.read_output_file(file_OPT)
    out_PHO = stout.read_output_file(file_PHO)

    #z = out_OPT['keywords']['PHO_Redshift']

    #Plot the fit without photometry:
    stplot.plot_spec_simple(out_OPT, ax=axis
                     , syn_color='r', syn_label=label2
                     , plot_error=False, w0_color='y', PHO_color='gold', PHO_edgecolor='r', PHO_markersize=7)

    #Plot fit with photometric constraints:
    stplot.plot_spec_simple(out_PHO, ax=axis
                    , plot_obs=False, syn_label=label1
                    , plot_error=False, PHO_edgecolor='b', PHO_markersize=7)

    #z = out_PHO['keywords']['PHO_Redshift']

    #fluxes = [wololo.abmagstoflux_wlpiv(totalmags[i], out_PHO['PHO']['PivotLamb'][i])*(1+z)/out_PHO['keywords']['fobs_norm'] for i in range(2)]

    #plt.plot(out_PHO['PHO']['MeanLamb']/(1+z), fluxes, '^m', label=r'$O_{\mathrm{PHO}}^{Tot}$', zorder=10)

    ##Is also interesting to plot the filter files (shifted to restframe):
    #plot_filter('./filters/NUV.dat', ax=axis, redshift=z)
    #plot_filter('./filters/FUV.dat', ax=axis, redshift=z)

    #Making room for legend above the spectra:
    #axis.set_ylim(0,3)

    #plt.legend(ncol=2)





def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def data2figcoords(data_value, fig_edges, data_limits):
    '''
    data_value:  value to be converted to fig coordinates
    fig_edges:   [bottom, top] for y coordinates, [left, right] for x coordinates
    data_limits: x_lim or y_lim
    '''

    data_value_normed = np.abs((data_value-data_limits[0])/(data_limits[1]-data_limits[0]))
    fig_coords        = data_value_normed*(fig_edges[1]-fig_edges[0]) + fig_edges[0]

    return fig_coords



def plot_embedded_SFH(x_main, y_main, x_emb, y_emb, x_bins, y_bins, x_lim, y_lim
, main_axlabels=None, top=0.98, bottom=0.08, left=0.07, right=0.98, main_color='k'
, emb_color=0.8, emb_ls='-', emb_axlabels=['',''], emb_label=None, plot_main=True
, fig=None, labeled_subplot_index=1):
    '''
    x_bins: list of lists, example [[-22,-21],[-21,-20]]
    y_bins: list of lists, example [[3,4,5],[2,3,3.5,4]]
    '''

    ax_list = []

    if fig==None:
        fig=plt.figure()

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    if plot_main==True:

        plt.plot(x_main, y_main, '.', color=main_color, ms=1)
        plt.xlim(x_lim)
        plt.ylim(y_lim)

        if main_axlabels!=None:
            plt.xlabel(main_axlabels[0])
            plt.ylabel(main_axlabels[1])

    #The main code, create subplots
    for i in range(len(x_bins)):

        x_flags, x_edges = bin_data(x_main, nbins=x_bins[i])
        y_flags, y_edges = bin_data(y_main, nbins=y_bins[i])

        for j in range(len(y_bins[i])-1):
            x_low = data2figcoords(x_edges[0], [left, right], x_lim)
            x_upp = data2figcoords(x_edges[1], [left, right], x_lim)
            y_low = data2figcoords(y_edges[j], [bottom, top], y_lim)
            y_upp = data2figcoords(y_edges[j+1], [bottom, top], y_lim)

            ax = plt.axes([x_low, y_low, x_upp-x_low, y_upp-y_low])

            ax.plot(np.log10(x_emb), np.mean(y_emb[x_flags[0]&y_flags[j]], axis=0), color=emb_color, ls=emb_ls, label=emb_label)

            ax_list.append(ax)


    #Remove labels from all but one axis and add labels:
    for i in range(len(ax_list)):
        if i!=labeled_subplot_index:
            ax_list[i].tick_params(axis='both', labelleft='off', labelbottom='off')
        else:
            ax_list[i].set_ylabel(emb_axlabels[0])
            ax_list[i].set_ylabel(emb_axlabels[1])


    return ax_list
