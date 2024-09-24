#!/usr/bin/env python3
'''
@file gpr.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 1, 2024
@brief Postscripts for running the GPR V+jets fit


This runs the GPR fit to a single variable in one channel. It uses as inputs the
`{var}__v__fatjet_m` histograms in the reader, which are filled with the inclusive SR+MCR.
The script will run a fit for each bin specified in [get_bins_y], which includes the full
contour scan to obtain the marginalized posterior. In general, see the CONFIG section for
some hardcodes.

By default, the fit will be run using the event subtraction scheme, `vjets = data - ttbar
- stop - diboson`. Set the signal strength parameters --mu-stop and --mu-ttbar if you want
to scale the respective samples. If you supply the `--closure-test` flag, will fit the GPR
to just the MC V+jets sample instead.

Results are saved into a file `gpr_fit_results.csv` containing both a 2sigma contour scan
and the simple MMLE fit, and a ROOT histogram in `gpr_{lep}_{var}_vjets_yield.root` which
can be input into ResonanceFinder. Note that the histogram file can be created from the
CSV directly without rerunning the fits using the `--from-csv-only` option, if you need to
edit or rerun a specific bin.

The script will generate a plot named `gpr_{lep}_{var}_summary` in both png and pdf
formats containing a summary distribution of the fits. For each bin fit, will also
generate the following plots:
    - cont_nlml: contour lines of the NLML space as a function of the two hyperparameters.
    - cont_yields: fitted SR yields in the same space.
    - fit_opt: posterior prediction using the MMLE fit
    - fit_m1s/fit_p1s: posterior predictions using +-1 sigma in the hyperparameter space
      fits
    - fit_pm1s: the above 3 posterior means superimposed onto the same plot
    - p_intr: the marginalized posterior distribution

-----------------------------------------------------------------------------------------
SETUP
-----------------------------------------------------------------------------------------

    setupATLAS
    lsetup "root recommended"
    lsetup "python centos7-3.9"

Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(

------------------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------------------
Check the hardcoded binnings and other options in the CONFIG block near the bottom.

Check [utils.Sample] and [utils.Variable] to make sure the hardcoded naming stuctures are
correct.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    gpr.py filepath/formatter_1.root [...] \
        --lepton 1 \
        --var vv_m \
        [OPTIONAL FLAGS]

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel number, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}lep/{sample}.root
    
See [utils.FileManager] for details.

------------------------------------------------------------------------------------------
IMPLEMENTATION DETAILS
------------------------------------------------------------------------------------------
There's a lot of code here so here's a quick breakdown of the main callstack:

    1. main() - Fetches files and defines the config from command line args
    2. run() - Actual run function, which can be called from another script too. Takes a
       config and runs the full set of fits for the given variable. Handles fetching
       histograms and prepping the data.
    3. gpr_likelihood_contours() - Run wrapper for a single contour scan (i.e. one bin in 
       the above variable). Handles saving results and plots of the scan.
    4. ContourScanner - Class that actually handles the contour scan and calculating the
       marginal posterior.
    5. GPR - Class that handles a single GPR fit (which is repeated 25x25 times in the 
       contour scan).

'''

from __future__ import annotations
from typing import Union

# WARNING! The sklearn imports seem to cause segfaults with ROOT, even when they're not
# used. However they don't always happen, so you can just rerun until it works.
import ROOT # type: ignore
ROOT.gROOT.SetBatch(ROOT.kTRUE)
ROOT.TH1.SetDefaultSumw2(ROOT.kTRUE)

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing

from plotting import plot
import utils


###############################################################################
###                                  UTILS                                  ###
###############################################################################

def normal_cum(sigmas):
    '''
    Returns the fraction of events of a normal distribution within [sigmas] standard 
    deviations from the mean.

    1 => 0.6826894921370859
    2 => 0.9544997361036416
    3 => 0.9973002039367398
    '''
    return 1 - 2 * stats.norm.sf(sigmas) # sf is one-sided so need to double

def find_percentile_range(h, sigmas, x_values=False):
    '''
    Assumes [h] is a probability distribution that sums to 1 and has a singular peak.
    Finds the range of x values around the peak within a given percentile, passed as a
    z-score in [sigmas].

    This function generally assumes a uniform distirubtion inside each bin. 

    @return mode, bin_range
        mode is the bin index of the mode. bin_range is a pair of floats in bin-index space, 
        containing the (lower, upper) bounds of the range. Since the interval will not align 
        with the bins, a fractional part is returned. Will return None if the range is not
        contained in the histogram. Note these are 1-indexed, so 1.0 refers the to the left
        edge of the first bin.
    '''
    ### Get mode ###
    max_bin = 0
    max_val = 0
    for i in range(1, h.GetNbinsX() + 1):
        p = h.GetBinContent(i)
        if p > max_val:
            max_val = p
            max_bin = i
    cum_remaining = normal_cum(sigmas) - max_val

    ### Error check ###
    if max_bin == 1 or max_bin == h.GetNbinsX():
        plot.warning("find_percentile_range() Mode is at end of histogram range.")
        cum_remaining /= 2

    ### Find bins moving out from peak ###
    bin_range = [None, None] # float bounds for output
    bin_hi = max_bin + 1 # exclusive bins (i.e. next bins to look at)
    bin_lo = max_bin - 1
    while cum_remaining > 0:
        p_hi = h.GetBinContent(bin_hi) if bin_hi <= h.GetNbinsX() else 0
        p_lo = h.GetBinContent(bin_lo) if bin_lo > 0 else 0
        if p_hi <= 0 and p_lo <= 0: 
            break
        elif p_hi > p_lo:
            if p_hi >= cum_remaining:
                bin_range[1] = bin_hi + cum_remaining / p_hi
                break
            else:
                cum_remaining -= p_hi
                bin_range[1] = bin_hi + 1
                bin_hi += 1
                if bin_hi > h.GetNbinsX():
                    plot.warning("find_percentile_range() Range out-of-bounds, will continue assuming symmetric distribution.")
                    bin_range[1] = None
                    cum_remaining /= 2
        else:
            if p_lo >= cum_remaining:
                bin_range[0] = bin_lo + 1 - cum_remaining / p_lo
                break
            else:
                cum_remaining -= p_lo
                bin_range[0] = bin_lo
                bin_lo -= 1
                if bin_lo <= 0:
                    plot.warning("find_percentile_range() Range out-of-bounds, will continue assuming symmetric distribution.")
                    bin_range[0] = None
                    cum_remaining /= 2

    if x_values:
        x_mid = h.GetBinCenter(max_bin)
        if bin_range[0] is not None:
            bin_low = int(bin_range[0])
            x_min = h.GetBinLowEdge(bin_low) + (bin_range[0] - bin_low) * h.GetBinWidth(bin_low)
        else:
            x_min = None
        if bin_range[1] is not None:
            bin_hi = int(bin_range[1])
            x_max = h.GetBinLowEdge(bin_hi) + (bin_range[1] - bin_hi) * h.GetBinWidth(bin_hi)
        else:
            x_max = None
        return x_mid, x_min, x_max
    else:
        return max_bin, bin_range

def linspace_bin_centered(start, end, n):
    '''
    Returns the bin centers of [n] equally space bins between [start, end].
    '''
    width = end - start
    offset = width / (2 * n)
    return np.linspace(start + offset, end - offset, n)

def get_chi2_quantile(sigmas, ndof):
    '''
    Returns the chi2 distibution cutoff for a given quantile.

    @param sigmas
        The number of standard deviations. I.e. sigmas=2 for a 95% CI.
    '''
    return stats.chi2.ppf(normal_cum(sigmas), ndof)

def get_sr_cr(h, sr_def):
    h_sr = h.Clone()
    h_cr = h.Clone()
    for x in range(1, h.GetNbinsX() + 1):
        low = h.GetBinLowEdge(x)
        high = h.GetBinLowEdge(x + 1)
        if high > sr_def[0] and low < sr_def[1]:
            h_cr.SetBinContent(x, 0)
            h_cr.SetBinError(x, 0)
        else:
            h_sr.SetBinContent(x, 0)
            h_sr.SetBinError(x, 0)
    return h_sr, h_cr

def get_bin_range(h, xmin, xmax) -> tuple[int, int]:
    i_min = None
    i_max = None
    for x in range(1, h.GetNbinsX() + 2):
        low = h.GetBinLowEdge(x)
        if low == xmax:
            i_max = x
            break
        if low == xmin:
            i_min = x
    if not i_min:
        raise RuntimeError(f"get_bin_range() couldn't find bin matching x={xmin}")
    if not i_max:
        raise RuntimeError(f"get_bin_range() couldn't find bin matching x={xmax}")
    return (i_min, i_max)
            
class RatioWithError:
    def __init__(self, num, num_err, denom, denom_err):
        self.num = num
        self.num_err = num_err
        self.denom = denom
        self.denom_err = denom_err

        self.ratio = num / denom
        self.ratio_err_num = self.ratio * num_err / num
        self.ratio_err_denom = self.ratio * denom_err / denom
        self.ratio_err = (self.ratio_err_num**2 + self.ratio_err_denom**2) ** 0.5

    def title(self, num_label='', denom_label='', percent_diff=False, errors=True, decimals=None):
        if decimals is None:
            decimals = 1 if percent_diff else 3

        if percent_diff:
            ratio = 100 * (self.ratio - 1)
            scale = 100
        else:
            ratio = self.ratio
            scale = 1

        title = f'{ratio:.{decimals}f}'
        
        if errors:
            title += f' #pm {scale*self.ratio_err:.{decimals}f}'
            if num_label or denom_label:
                title += ' #splitline{#scale[0.8]{'
            if num_label:
                title += f'{scale*self.ratio_err_num:.{decimals}f} {num_label}'
                if not denom_label: title += ')'
            title += '}}{#scale[0.8]{'
            if denom_label:
                title += f'{scale*self.ratio_err_denom:.{decimals}f} {denom_label}'
            title += '}}'
        return title

class FitOverMCRatio(RatioWithError):
    def __init__(self, fit, fit_err, mc, mc_err):
        super().__init__(fit, fit_err, mc, mc_err)

    def title_ratio(self, fit_error_only=False, **kwargs):
        if fit_error_only:
            return f'SR fit/MC = {self.ratio:.2f} #pm {self.ratio_err_num:.2f}'
        else:
            return 'SR fit/MC = ' + super().title('fit', 'MC', **kwargs)

    def title_percent(self, **kwargs):
        return 'SR %#Delta_{fit-MC} = ' + super().title('fit', 'MC', percent_diff=True, **kwargs)

    def legend_percent(self):
        '''
        Condensed title for legends that shows the % difference (fit - MC) with just
        the fit error.
        '''
        ratio = 100 * (self.ratio - 1)
        error = 100 * self.ratio_err_num
        return f'%#Delta={ratio:.1f}#pm{error:.1f}'

def format_error(val, err):
    if val > 100:
        return f'{val:.0f} #pm {err:.0f}'
    else:
        return f'{val:.2f} #pm {err:.2f}'

def set_sqrtn_errors(h, width_scaled=False):
    for i in range(h.GetNcells()):
        val = h.GetBinContent(i)
        err = h.GetBinError(i)
        if val <= 0:
            h.SetBinContent(i, 0)
            h.SetBinError(i, 0)
        elif width_scaled:
            h.SetBinError(i, np.sqrt(val / h.GetBinWidth(i)))
        else:
            h.SetBinError(i, np.sqrt(val))

def clamp_errors(h, threshold):
    for i in range(h.GetNcells()):
        val = h.GetBinContent(i)
        err = h.GetBinError(i)
        if val > 0 and err > val * threshold:
            h.SetBinError(i, val * threshold)
            

def weighted_integral(h, x_min, x_max, weights, width_scaled=True):
    start, end = get_bin_range(h, x_min, x_max)
    assert(len(weights) == end - start)

    val = 0
    err = 0
    for i in range(end - start):
        if width_scaled:
            w = h.GetBinWidth(i + start)
            val += weights[i] * h[i + start] * w
            err += (weights[i] * h.GetBinError(i + start) * w)**2
        else:
            val += weights[i] * h[i + start]
            err += (weights[i] * h.GetBinError(i + start))**2
    return val, err**0.5

def calculate_signal_strength_weights(
        fitter : GPR, 
        h_diboson: ROOT.TH1F, 
        sr_window: tuple[float, float], 
        h_sr: ROOT.TH1F = None,
    ) -> tuple[list[float], tuple[float, float], ROOT.TH1F]:
    '''
    This calculates the signal strength of the diboson signal with shape taken from
    [h_diboson]. All histograms should be width scaled already.

    @param h_sr
        The data in the SR (doesn't need to be trimmed). If this is None, will assume
        asimov data  = gpr + diboson.
    @returns
        tuple[0]: A list of weights. These are prescaled, so all you have to do is mu =
        n_integral - gpr_integral.

        tuple[1]: The N_i integral (val, err). Remember we split the sum into n_i * w_i/s_i
        - b_i * w_i/s_i, so we calculate the first term here.

        tuple[2]: [h_sr] if it is not None, otherwise the created asimov data histogram.
    '''
    bin_range = get_bin_range(h_diboson, *sr_window)
    bin_centers = [h_diboson.GetBinCenter(i) for i in range(*bin_range)]
    gpr_preds = fitter.predict(bin_centers, return_std=True)

    ### Create asimov data = gpr (MMLE) + diboson ###
    if not h_sr:
        h_sr = h_diboson.Clone()
        for i in range(*bin_range):
            v = h_diboson.GetBinContent(i) + gpr_preds[0][i - bin_range[0]]
            h_sr.SetBinContent(i, v)
            h_sr.SetBinError(i, (v / h_diboson.GetBinWidth(i)) ** 0.5)
            # Estimate error as just sqrt(n) (but remember it's width scaled)
            # Real data will be larger but minus ttbar background
    
    ### Get the weights ###
    # Remember we split the sum into n_i * w_i/s_i - b_i * w_i/s_i. The first term is
    # calculated below. Also, remember everything is normalized by 1/sum(w_i)
    weights = []
    weight_sum = 0
    n_integral = 0
    n_integral_err = 0
    s_over_sqrtn = []
    for i in range(*bin_range):
        w = h_diboson.GetBinWidth(i)
        s = h_diboson.GetBinContent(i) * w
        se = h_diboson.GetBinError(i) * w
        n = h_sr.GetBinContent(i) * w
        ne = h_sr.GetBinError(i) * w
        b = gpr_preds[0][i - bin_range[0]] * w
        be = gpr_preds[1][i - bin_range[0]] * w

        mu = (n - b) / s
        var_num = (ne**2 + be**2) / (n - b)**2
        var_denom = se**2 / s**2
        var = mu**2 * (var_num + var_denom)

        weight = 1 / var
        weights.append(weight / s)
        weight_sum += weight

        ratio = n * weight / s
        n_integral += ratio
        n_integral_err += ratio**2 * (ne**2 / n**2 + se**2 / s**2) if n > 0 else 0

    weights = np.array(weights) / weight_sum
    return weights, (n_integral / weight_sum, n_integral_err**0.5 / weight_sum), h_sr


###############################################################################
###                                 RESULTS                                 ###
###############################################################################

class FitResults:
    '''
    This class handles saving and retrieving fit results from a CSV. If uses a pandas 
    dataframe as the intermediary in [self.df].
    '''
    index_cols = ['lep', 'vary', 'variation', 'fitter', 'bin']

    def __init__(self, filepath='fit_results.csv'):
        self.filepath = filepath
        try:
            self.df = pd.read_csv(filepath, index_col=self.index_cols, keep_default_na=False) #, dtype={'lep': str}
        except:
            self.df = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=self.index_cols), columns=['val', 'err_up', 'err_down'])

    def filter(self, lep=slice(None), vary=slice(None), variation=slice(None), fitter=slice(None), bin=slice(None), **_):
        return self.df.loc[(lep, vary, variation, fitter, bin), :]

    def contains(self, lep, vary, variation, fitter, bin, **_) -> bool:
        return self.df.index.isin([(lep, vary, variation, fitter, bin)]).any()

    def get_entry(self, lep, vary, variation, fitter, bin, unscale_width=False, **_) -> tuple[float, float, float]:
        if not isinstance(bin, str):
            bin_edges = bin
            bin = f'{bin[0]},{bin[1]}'
        elif unscale_width:
            bin_edges = [float(x) for x in bin.split(',')]

        out = self.df.loc[(lep, vary, variation, fitter, bin)]

        if unscale_width:
            w = bin_edges[1] - bin_edges[0]
            return out * w
        return out

    def set_entry(self, lep, vary, variation, fitter, bin, val, err_up, err_down, **_):
        self.df.loc[(lep, vary, variation, fitter, bin)] = [val, err_up, err_down]
        self.save()

    def save(self, filepath=None):
        if filepath is None: filepath = self.filepath
        self.df.sort_index(inplace=True)
        self.df.to_csv(filepath, float_format='%e')

    def get_histogram(self, lep, vary, variation, fitter, bins, histname):
        bins_float = np.array(bins, dtype=float)
        h = ROOT.TH1F(histname, histname, len(bins_float) - 1, bins_float)
        for i in range(len(bins) - 1):
            bin = f'{bins[i]},{bins[i+1]}'
            entry = self.get_entry(lep, vary, variation, fitter, bin, unscale_width=True)
            h.SetBinContent(i + 1, entry['val'])
            h.SetBinError(i + 1, (entry['err_up'] + entry['err_down']) / 2)
            # I don't think ResonanceFinder can handle a TGraphAsymmErrors so we have to
            # symmetrize the error
        return h

    def get_graph(self, lep, vary, variation, fitter, bins, unscale_width=False):
        g = ROOT.TGraphAsymmErrors(len(bins) - 1)
        for i in range(len(bins) - 1):
            bin = f'{bins[i]},{bins[i+1]}'
            width = bins[i+1] - bins[i] if unscale_width else 1
            y_mid = (bins[i+1] + bins[i]) / 2

            entry = self.get_entry(lep, vary, variation, fitter, bin)
            g.SetPoint(i, y_mid, entry['val'] * width)
            g.SetPointError(i, y_mid - bins[i], bins[i+1] - y_mid, entry['err_down'] * width, entry['err_up'] * width)
        return g




###############################################################################
###                                 PLOTTING                                ###
###############################################################################


def plot_gpr_fit(
        h_cr,
        gpr : GPR, 
        gpr_range, 
        h_sr=None, 
        gpr_sigmas=2,
        gpr_color=plot.colors.blue,
        **kwargs,
    ):
    '''
    Plots a single Gaussian process fit against an m(J) distribution.
        1. Data points are shown as error bars, with the CR in black and the SR in red.
        2. The GPR is shown as a blue band.

    @param h_cr
        Data point histogram
    @param h_sr
        Drawn as red points with label "Signal Region". Setting this option will cause
        the function to assume [h_cr] is MC, and change labels accordingly. Also, it
        will add a dashed band to indicate the sqrt(n) errors in the CR.
    @param gpr
        A scikit GaussianProcessRegressor object that has been fit to the data
    @param gpr_range
        A tuple of the x range to plot the gpr
    '''
    ### GPR ###
    g_gpr = gpr.create_graph(gpr_range, gpr_sigmas, gpr_color)

    ### Histogram styles ###
    h_cr.SetLineColor(ROOT.kBlack)
    h_cr.SetLineStyle(ROOT.kSolid)
    h_cr.SetMarkerSize(0)
    if h_sr is not None:
        h_sr.SetLineColor(plot.colors.red)
        h_sr.SetMarkerSize(0)

        h_sqrtn_errs = h_cr.Clone()
        set_sqrtn_errors(h_sqrtn_errs, width_scaled=True)
        h_sqrtn_errs.SetFillColor(plot.colors.gray)
        h_sqrtn_errs.SetFillStyle(3245)
        h_sqrtn_errs.SetLineWidth(0)
        h_sqrtn_errs.SetMarkerSize(0)

    ### Ratios ###
    X_ratios = []
    for i in range(1, h_cr.GetNbinsX() + 1):
        c = h_cr.GetBinCenter(i)
        if c > gpr_range[1]:
            break
        X_ratios.append(c)
    X_ratios = np.array(X_ratios, dtype=float).reshape((-1, 1))
    gpr_preds, _ = gpr.predict(X_ratios, return_std=True)
    # TODO use gpr error in ratio?

    ratio_hists = []
    ratio_opts = []
    for h,o in zip([h_cr, h_sr], ['E', 'E']):
        if h is None: continue
        h = h.Clone()
        for i in range(1, h.GetNbinsX() + 1):
            if i <= len(gpr_preds):
                h.SetBinContent(i, h.GetBinContent(i) / gpr_preds[i - 1])
                h.SetBinError(i, h.GetBinError(i) / gpr_preds[i - 1])
            else:
                h.SetBinContent(i, 0)
                h.SetBinError(i, 0)

        ratio_hists.append(h)
        ratio_opts.append(o)

    ### Collect plot histograms ###
    hists = [g_gpr, h_cr]
    plot_opts = ['3C', 'E']
    legend = [
        [h_cr, 'Control Region' if h_sr else 'Data', 'LE'],
        [g_gpr, f'GPR Mean #pm {gpr_sigmas}#sigma', 'LF'],
    ]
    if h_sr:
        hists.extend([h_sr, h_sqrtn_errs])
        plot_opts.extend(['E', 'E2'])
        legend.insert(-1, [h_sr, 'Signal Region', 'LE'])
        legend.insert(-1, [h_sqrtn_errs, '#sqrt{N} errors', 'F'])

    ### Draw ###
    kwargs.setdefault('text_pos', 'topright')
    kwargs.setdefault('legend', legend)
    kwargs.setdefault('ytitle2', ('MC' if h_sr else 'Data') + ' / GPR')
    kwargs.setdefault('xtitle', 'm(J) [GeV]')
    kwargs.setdefault('opts', plot_opts)
    kwargs.setdefault('opts2', ratio_opts)
    kwargs.setdefault('y_range', [0, None])
    kwargs.setdefault('y_range2', [0.5, 1.5])
    kwargs.setdefault('hline', {'y':1, 'style':ROOT.kDashed})
    kwargs.setdefault('outlier_arrows', False)
    plot.plot_ratio(hists, ratio_hists, **kwargs)

    return g_gpr


def plot_nlml_contours(h_nlml, min_nlml, marker, **kwargs):
    '''
    2D contour plot of the marginal likelihood.

    @param min_nlml
        The minimum of [h_nlml], used for determining the contours.
    @param marker
        A ROOT marker located at the minimum point.
    '''
    ### CI contours ###
    h_68 = h_nlml.Clone()
    h_68.SetLineColor(plot.colors.red)
    h_68.SetContour(1, np.array(min_nlml + get_chi2_quantile(1, 2) / 2))

    h_95 = h_68.Clone()
    h_95.SetLineStyle(ROOT.kDashed)
    h_95.SetContour(1, np.array(min_nlml + get_chi2_quantile(2, 2) / 2))

    objs = [h_nlml, h_68, h_95]

    ### Opts ###
    kwargs['textpos'] = 'backward diagonal'
    kwargs['legend'] = [
        (h_68, '1#sigma CI', 'L'),
        (h_95, '2#sigma CI', 'L'),
    ]
    kwargs['ztitle'] = 'Negative log marginal likelihood'
    kwargs['ytitle'] = 'Gamma Factor'
    kwargs['xtitle'] = 'Length Scale [GeV]'

    kwargs['opts'] = ['CONT1Z', 'CONT3', 'CONT3']
    kwargs['right_margin'] = 0.25
    kwargs['title_offset_z'] = 1.5
    
    kwargs['logx'] = True
    kwargs['logy'] = True
    kwargs['x_range'] = None
    kwargs['y_range'] = None

    def callback(plotter):
        marker.Draw()

    plot.plot(objs, callback=callback, **kwargs)


def plot_yield_scan(h_yields, marker, **kwargs):
    '''
    2D surf plot of the fitted yields as a function of the hyperparameters.

    @param marker
        A ROOT marker located at the minimum point.
    '''
    kwargs['textpos'] = 'backward diagonal'
    kwargs['legend'] = None
    kwargs['ztitle'] = 'SR fit yield'
    kwargs['ytitle'] = 'Gamma Factor'
    kwargs['xtitle'] = 'Length Scale [GeV]'
    kwargs['zdivs'] = 6

    kwargs['opts'] = ['COLZ']
    kwargs['right_margin'] = 0.25
    kwargs['title_offset_z'] = 1.5

    kwargs['logx'] = True
    kwargs['logy'] = True
    kwargs['x_range'] = None
    kwargs['y_range'] = None
    kwargs['z_range'] = (None, None)

    def callback(plotter):
        marker.Draw()

    plot.plot([h_yields], callback=callback, **kwargs)


def plot_pf(
        h_pf, 
        filename='pf.png', 
        subtitle=[], 
        mc_yield : tuple[float, float] = None,
        **plot_opts,
    ):
    '''
    Creates a graph of p(f | X,y), where f is the yield in the SR region.

    @param h_pf
        Histogram of the probabilities. See [find_percentile_range].
    @param x_scale
        Multiplies all x values in [h_pf] by this value.
    @param mc_yield
        Tuple (value, error) of the MC yield. Overlays a point with horizontal errors bars
        to indicate the MC yield.
    @return mode, -1sigma, +1sigma
    '''
    c = ROOT.TCanvas('c1', 'c1', 1000, 800)

    ### Draw fill boxes for 1sigma CI ###
    boxes = []
    box_style = ROOT.TBox(0, 0, 1, 1)
    box_style.SetFillColor(plot.colors.pastel_blue)
    box_style.SetLineWidth(0)

    ### Get x values on the way ###
    max_bin, bin_range = find_percentile_range(h_pf, 1)
    lower = bin_range[0] or 0
    upper = bin_range[1] or h_pf.GetNbinsX() + 1
    bin_lower = int(lower)
    bin_upper = int(upper)
    x_mid = h_pf.GetXaxis().GetBinCenter(max_bin)
    x_lower = None # These can be out-of-bounds of h_pf
    x_upper = None

    for i in range(bin_lower, bin_upper + 1):
        x1 = h_pf.GetXaxis().GetBinLowEdge(i)
        x2 = h_pf.GetXaxis().GetBinLowEdge(i + 1)
        if i == bin_lower:
            x1 = x1 + (x2 - x1) * (lower - bin_lower)
            x_lower = x1
        elif i == bin_upper:
            x2 = x1 + (x2 - x1) * (upper - bin_upper)
            x_upper = x2

        b = box_style.Clone()
        b.SetX1(x1); b.SetX2(x2); b.SetY1(0); b.SetY2(h_pf.GetBinContent(i))
        boxes.append(b)

    ### Set frame, with y_range ###
    digits = 0 if x_mid > 100 else 1
    opts = dict(
        text_pos='topleft',
        subtitle=[
            *subtitle,
            f'f = {x_mid:.{digits}f} ^{{+{x_upper - x_mid:.{digits}f}}}_{{#minus{x_mid - x_lower:.{digits}f}}}',
        ],
        ytitle='p(f | X,y)',
        xtitle='f #equiv SR Yield',
        ydivs=506,
        y_range=[0, None],
    )
    opts.update(plot_opts)
    plotter = plot.Plotter(c, **opts)

    ### Boxes ###
    plotter.add_primitives(boxes)

    ### Legend ###
    legend = [(box_style, '1#sigma CI', 'F')]
    if mc_yield:
        g_mc_err = ROOT.TGraphErrors(1)
        legend += [(g_mc_err, 'MC Yield', 'PL')]

    ### Mode line and MC error ###
    l = ROOT.TLine(x_mid, 0, x_mid, h_pf.GetBinContent(max_bin))
    l.SetLineStyle(ROOT.kDashed)
    plotter.add_primitives([l])

    ### Main histogram ###
    plotter.add([h_pf],
        opts='HIST', 
        markersize=0,
        legend=legend,
    )

    ### MC yield ###
    plotter.compile() # get y_range
    if mc_yield:
        g_mc_err.SetPoint(0, mc_yield[0], .07 * plotter.y_range[1])
        g_mc_err.SetPointError(0, mc_yield[1], 0)
        plotter.add_primitives([g_mc_err], 'SAME P')

    ### Draw ###
    plotter.draw()
    plot.save_canvas(c, filename)
    return x_mid, x_lower, x_upper


def plot_updown_fits(
        scanner : ContourScanner, 
        h_cr=None, 
        h_mc=None, 
        sr_window: tuple[float, float]=None,
        filename='{stub}',
        subtitle=[],
        **plot_opts,
    ):
    '''
    Plots the GPR posterior at the min nlml hyperparameter values and +- 1 sigma values.
    Makes 4 plots:
        * A summary plot with all 3 posterior means
        * 3 individual fit plots for each hyperparameter setting
    
    @param h_cr
        The main histogram that was fitted. [h_cr] should have been scaled by bin widths.
    @param h_mc
        If [h_cr] was derived from MC, pass instead this option containing the full MC 
        histogram, which will be plotted instead, with the full closure test. This replaces
        [h_cr], which will be unused. In this case, pass also [sr_window].
    @param subtitle
        This function will add two lines:
            1. Kernel description
            2. Closure as a % difference, if [mc_sr_yield] is not None, otherwise the 
               total yield.
    @param filename
        This string will be formatted with a field {stub} for the different plots.
    '''
    ### MC parsing ###
    if h_mc:
        assert(sr_window is not None)
        h_sr, h_cr = get_sr_cr(h_mc, sr_window)
        mc_sr_yield = plot.integral_user(h_mc, sr_window, use_width=True, return_error=True)
    else:
        h_sr = None
        mc_sr_yield = None

    ### Individually ###
    gpr_graphs = []
    gpr_legend = []
    for name, arg, legend, color in zip(
        ['m1s', 'p1s', 'opt'],
        ['min_scale', 'max_scale', 'optimal'],
        ['-1#sigma', '+1#sigma', 'Opt'],
        [plot.colors.green, plot.colors.purple, plot.colors.blue],
    ):
        ### Refit ###
        theta = getattr(scanner, arg + '_theta')
        integral = getattr(scanner, arg + '_int')
        scanner.fitter.refit(theta)

        ### Opts ###
        opts = plot_opts.copy()
        opts['subtitle'] = [
            *subtitle,
            f'{scanner.fitter}',
        ]
        if h_mc:
            intr = FitOverMCRatio(*integral, *mc_sr_yield)
            opts['subtitle'].append(
                'SR %#Delta_{fit-mc} = ' + intr.title('fit', 'mc', True),
            )
        else:
            opts['subtitle'].append(
                'SR Yield = ' + format_error(*integral),
            )

        ### Graph and fit ###
        g = plot_gpr_fit(
            h_cr=h_cr, 
            h_sr=h_sr,
            gpr=scanner.fitter, 
            gpr_range=scanner.fitter.fit_range,
            gpr_color=color,
            filename=filename.format(stub = f'fit_{name}'),
            **opts,
        )
        gpr_graphs.append(g)

        ### Legend for combined graph below ###
        if h_mc:
            legend_label = f'{legend} (l={theta[1]:.0f}, {intr.legend_percent()})'
        else:
            legend_label = f'{legend} (l={theta[1]:.0f})'
        gpr_legend.append((g, legend_label, 'L'))

    ### Reset to avoid side effects ###
    scanner.fitter.refit(scanner.scikit_theta)

    ### Summary plot ###
    hists = [*gpr_graphs, h_cr]
    h_cr.SetLineColor(ROOT.kBlack)
    opts = plot_opts.copy()
    opts.update({
        'opts': ['CX'] * 3 + ['E'],
        'filename': filename.format(stub = 'fit_pm1s'),
        'text_pos': 'topright',
        'markersize': 0,
        'subtitle': subtitle,
        'y_range': [0, None],
    })
    if h_mc:
        hists += [h_sr]
        opts['opts'] += ['E']
        h_sr.SetLineColor(plot.colors.red)
        opts['legend'] = [
            [h_cr, 'Control Region', 'LE'],
            [h_sr, 'Signal Region', 'LE'],
            *gpr_legend,
        ]
    else:
        opts['legend'] = [
            [h_cr, 'Data', 'LE'],
            *gpr_legend,
        ]

    plot.plot(hists, **opts)


def _fractional_uncertainties(hists, opts, postfix='3', percents=False):
    '''
    Ratio callback that returns histograms for each input histo where the bin value is 
    the fractional uncertainty (error / value).

    Since the errors are 0, make sure to plot with markersize > 0 or HIST mode. Updates
    [opts] with some default plotting options.
    '''
    mult = 100 if percents else 1
    rs = []
    for h in hists:
        if 'TH' in h.ClassName():
            h = h.Clone()
            for i in range(h.GetNbinsX() + 2):
                v = h.GetBinContent(i)
                if v > 0:
                    h.SetBinContent(i, mult * h.GetBinError(i) / v)
                else:
                    h.SetBinContent(i, 0)
                h.SetBinError(i, 0)
            rs.append(h)
        else:
            h = h.Clone()
            for i in range(h.GetN()):
                v = h.GetPointY(i)
                if v > 0:
                    h.SetPointY(i, mult * h.GetErrorY(i) / v)
                else:
                    h.SetPointY(i, 0)
                h.SetPointEYlow(i, 0)
                h.SetPointEYhigh(i, 0)
            rs.append(h)

    opts['ytitle' + postfix] = ('%' if percents else 'Frac.') + ' Uncer.'
    opts['ignore_outliers_y' + postfix] = 0
    opts['opts' + postfix] = 'HIST'
    opts['linecolor' + postfix] = opts.get('linecolor', plot.colors.tableu)
    return rs


def plot_summary_distribution(hists, 
        subplot2='ratios',
        subplot3='errors',
        ratio_denom=lambda i: 0,
        **kwargs
    ):
    '''
    Creates a plot of the full fitted distribution, like m(VV), using discrete bins.
    Creates two subplots, the first shows the Fit / MC ratio, and the second shows the
    percent uncertainty.

    In [kwargs] you can pass additional options to [plot.plot_discrete_bins]. Most
    notably, you will probably want to specify 'filename' and 'edge_labels'.
    
    @param subplot2, subplot3
        Subplot specifications. Can be either
            * 'ratios', which will plot the ratio to the histogram given by [ratio_denom].
            * 'errors', which will plot the relative % error of each point.
        Or a function f(hists, **kwargs) -> new_hists. Set to None to omit.
    @param ratio_denom 
        A function (i_hist) -> i_denom that given an index into [hists], returns the index
        of the histogram that should be used as the denominator in the 'ratios' plot. If
        i_denom == i_hist then omits this histogram from the ratio plot.
    '''

    ### Style ###
    # Do this before cloning for the ratio plots
    for i,h in enumerate(hists):
        h.SetLineWidth(2)
        h.SetLineColor(plot.colors.tableu(i))
        h.SetFillColorAlpha(plot.colors.tableu(i), 0.2)
        h.SetMarkerColor(plot.colors.tableu(i))
        h.SetMarkerStyle(ROOT.kFullCircle + i)
        # h.SetMarkerSize(0.05)

    ### Options ###
    kwargs.setdefault('opts', 'P2+')
    kwargs.setdefault('legend_opts', 'PE')
    kwargs.setdefault('ytitle', 'Events')
    kwargs.setdefault('logy', True)

    ### Ratio subplot ###
    if subplot2 == 'ratios' or subplot3 == 'ratios':
        _hists = []
        for i,h in enumerate(hists):
            i_denom = ratio_denom(i)
            if i == i_denom or i_denom is None: continue
            h_denom = hists[i_denom]

            r = h.Clone()
            if 'TGraph' in r.ClassName():
                r = plot.graph_divide(r, h_denom)
            else:
                r.Divide(h_denom)
            _hists.append(r)

        if subplot2 == 'ratios':
            postfix = '2'
            hists2 = _hists
        else:
            postfix = '3'
            hists3 = _hists
        kwargs.setdefault('opts' + postfix, 'P2+')
        kwargs.setdefault('ytitle' + postfix, '#frac{Fit}{MC}')
        kwargs.setdefault('ignore_outliers_y' + postfix, 0)
        kwargs.setdefault('hline' + postfix, {'y': 1, 'style': ROOT.kDashed})
        kwargs.setdefault('legend' + postfix, None)
    elif callable(subplot2):
        hists2 = subplot2(hists, **kwargs)
    else:
        hists2 = subplot2
    
    ### Fractional uncertainty ###
    if subplot2 == 'errors' or subplot3 == 'errors':
        _hists = _fractional_uncertainties(hists, {}, percents=True)
        if subplot2 == 'errors':
            postfix = '2'
            hists2 = _hists
        else:
            postfix = '3'
            hists3 = _hists
        kwargs.setdefault('opts' + postfix, 'P2+')
        kwargs.setdefault('ytitle' + postfix, '% Error')
        kwargs.setdefault('y_range' + postfix, [0, None])
        kwargs.setdefault('ignore_outliers_y' + postfix, 0)
    elif callable(subplot3):
        hists3 = subplot3(hists, **kwargs)
    else:
        hists3 = subplot3

    ### Plot ###
    if subplot3 is not None:
        def callback(*args):
            args[0].frame.GetYaxis().ChangeLabel(1, -1, 0)
        kwargs.setdefault('title_offset_x', 1)
        plot.plot_discrete_bins(hists, hists2, hists3, plotter=plot.plot_ratio3, callback=callback, **kwargs)
    elif subplot2 is not None:
        plot.plot_discrete_bins(hists, hists2, plotter=plot.plot_ratio, **kwargs)
    else:
        plot.plot_discrete_bins(hists, plotter=plot.plot, **kwargs)


def plot_postfit(h_gpr, h_data, h_ttbar, h_stop, h_diboson, filename, sr_window, mu_diboson=1, **plot_args):
    ### Hists ###
    h_diboson = h_diboson.Clone()
    h_diboson.Scale(mu_diboson)

    h_sum = h_gpr.Clone()
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)
    h_sum.Add(h_diboson)

    ratio = h_data.Clone()
    ratio.Divide(h_sum)

    h_sr, h_cr = get_sr_cr(h_data, sr_window)
    ratio_sr, ratio_cr = get_sr_cr(ratio, sr_window)

    ### Plot ###
    pads = plot.RatioPads(**plot_args)
    plotter1 = pads.make_plotter1(
        ytitle='Events / GeV',
        x_range=[50, 250],
        **plot_args,
    )
    plotter1.add(
        objs=[h_gpr, h_stop, h_ttbar, h_diboson], 
        legend=['GPR MMLE (V+jets)', 'Single top', 't#bar{t}', f'diboson (#mu={round(mu_diboson, 2)})'],
        stack=True,
        opts='HIST',
        fillcolor=plot.colors.pastel,
        linewidth=1,
    )
    plotter1.add(
        objs=[h_sum],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend_opts='F',
        legend=['GPR err + MC stat'],
    )
    plotter1.add(
        objs=[h_sr, h_cr],
        legend=['Data SR', 'Data MCR'],
        linecolor=[plot.colors.blue, ROOT.kBlack],
        markercolor=[plot.colors.blue, ROOT.kBlack],
        opts='PE',
        legend_opts='PEL',
    )
    
    plotter1.draw()

    ### Subplot ###
    plotter2 = pads.make_plotter2(
        ytitle='Data / Fit',
        xtitle='m(J) [GeV]',
        x_range=[50, 250],
    )
    plotter2.add(
        objs=[ratio_sr, ratio_cr],
        linecolor=[plot.colors.blue, ROOT.kBlack],
        markercolor=[plot.colors.blue, ROOT.kBlack],
        opts='PE',
        legend=None,
    )
    plotter2.draw()
    plotter2.draw_hline(1, ROOT.kDashed)

    plot.save_canvas(pads.c, filename)




###############################################################################
###                                  FITTER                                 ###
###############################################################################

class GPR:
    '''
    @property version
        String identifier of the kernel version
    @property gpr
        The GaussianProcessRegressor instance
    @property kernel
        The initial kernel used
    @property X_train, y_train, e_train
        The training data unmodified. Note that self.gpr.y_train_ contains the normalized
        data. self.gpr.X_train_ might be modified too?
    '''

    def __init__(self, version):
        self.version = version
        self.title = 'GPR'
        self.length_scale_bounds = (30, 1e3)
        self.noise_level_bounds = (1e-1, 1e2)
        self.use_data_alpha = True
        if version == 'rbf':
            rbf = RBF(length_scale=100, length_scale_bounds=self.length_scale_bounds)
            constant = ConstantKernel(1, self.noise_level_bounds)
            self.kernel = constant * rbf
        elif version == 'rbf+white':
            self.use_data_alpha = False
            self.noise_level_bounds = (1e-5, 1e2)
            rbf = RBF(length_scale=100, length_scale_bounds=self.length_scale_bounds)
            noise = WhiteKernel(noise_level=1, noise_level_bounds=self.noise_level_bounds)
            self.kernel = noise + rbf
        elif version == 'matern2.5':
            matern = Matern(length_scale=100, length_scale_bounds=self.length_scale_bounds, nu=2.5)
            constant = ConstantKernel(1, self.noise_level_bounds)
            self.kernel = constant * matern
        else:
            raise RuntimeError(f'Unknown version {version}')

    def __str__(self):
        if self.version == 'rbf+white':
            theta = np.exp(self.gpr.kernel_.theta)
            return f'{theta[0]**0.5:.2f}' + '^{2}#delta_{ij}' + f' + RBF(l={theta[1]:.0f})'
        else: 
            return f'{self.gpr.kernel_}' \
                .replace('length_scale', 'l') \
                .replace('nu', '#nu')
        


    def fit(self, h, fit_range):
        ### Training data ###
        X_train = []
        y_train = []
        e_train = []
        for i in range(1, h.GetNbinsX() + 1):
            x = h.GetBinCenter(i)
            if x < fit_range[0]: continue
            if x > fit_range[1]: break
            if h.GetBinContent(i) <= 0: continue
            X_train.append(x)
            y_train.append(h.GetBinContent(i))
            e_train.append(h.GetBinError(i))

        self.fit_range = fit_range
        self.X_train = np.reshape(X_train, (-1, 1))
        self.y_train = np.array(y_train)
        self.e_train = np.array(e_train) 
        scaler = preprocessing.StandardScaler().fit(self.y_train.reshape(-1, 1))

        alpha = (self.e_train / scaler.scale_)**2
        alpha_max = 0.2
        alpha[alpha > alpha_max] = 2 / (1/alpha[alpha > alpha_max] + 1/alpha_max) # Harmonic mean

        ### Gaussian process ###
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel, 
            normalize_y=True, 
            alpha=alpha if self.use_data_alpha else 1e-10, 
            n_restarts_optimizer=4,
        )
        self.gpr.fit(self.X_train, self.y_train)
        self._calculate_chi2()

    def refit(self, theta):
        '''
        Refits to the training data given a manual theta. Pass the non-log theta.

        Setting theta in grp.log_marginal_likelihood() saves the theta vector but scikit-
        learn doesn't update the cached parameters like self.L_ or self.alpha_, so need to
        recall fit().
        '''
        self.gpr.optimizer = None
        self.gpr.kernel.theta = np.log(theta)
        self.gpr.fit(self.X_train, self.y_train)

    def integral(self, sr_range, test_points=100):
        ### Split into [test_points] bins, and get the bin centers ###
        width = sr_range[1] - sr_range[0]
        offset = width / (2 * test_points)
        X = np.linspace(sr_range[0] + offset, sr_range[1] - offset, test_points).reshape((-1, 1))

        ### Prediction ###
        # From https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Linear_combinations
        # The variance on f = a * x is sigma_f^2 = a^T * Sigma_x * a
        # But we're just summing here so a = (1, 1, ...), and sigma_f^2 = sum(Sigma_x)
        mean, cov = self.gpr.predict(X, return_cov=True) # (n), (n, n)
        total = np.sum(mean) * width / test_points
        err = np.sqrt(np.sum(cov)) * width / test_points
        return total, err
    
    def weighted_integral(self, sr_range, weights, sample_points=3):
        '''
        As above, but apply a set of weights to each test point. The integral will be
        conducted using len(weights) * [sample_points] evenly spaced test points.
        '''
        width = sr_range[1] - sr_range[0]
        weights = np.repeat(weights, sample_points)
        n = len(weights)

        X = linspace_bin_centered(*sr_range, n).reshape((-1, 1))
        mean, cov = self.gpr.predict(X, return_cov=True) 
        
        total = mean.dot(weights) * width / n
        err = weights.dot(cov).dot(weights)
        err = np.sqrt(err) * width / n
        return total, err

    def _predict_single(self, x, return_std):
        X = [[x]]
        val, std = self.gpr.predict(X, return_std=True)
        if return_std:
            return val[0], std[0]
        else:
            return val[0]

    def predict(self, x, return_std=False):
        if isinstance(x, (float, int)):
            return self._predict_single(x, return_std)
        X = np.reshape(x, (-1, 1))
        return self.gpr.predict(X, return_std=True)
        
    def _calculate_chi2(self):
        self.ndof = len(self.y_train) - self.gpr.kernel_.n_dims # is this right?
        if self.ndof <= 0:
            self.chi2 = 0
            return
        
        vals, std = self.gpr.predict(self.X_train, return_std=True)
        accum = 0
        for y1,y2,e1 in zip(self.y_train, vals, self.e_train):
            accum += (y1 - y2)**2 / e1**2 # should use error on GPR?
        self.chi2 = accum / self.ndof

    def create_graph(self, range, sigmas=2, color=plot.colors.blue):
        X = linspace_bin_centered(*range, 100).reshape((-1, 1))
        mean, std = self.predict(X, return_std=True) 

        g_gpr = ROOT.TGraphErrors(len(X), X.reshape(-1), mean, np.zeros(mean.shape), sigmas * std)
        g_gpr.SetFillColorAlpha(color, 0.3)
        g_gpr.SetLineWidth(3)
        g_gpr.SetLineColor(color)

        return g_gpr
    
    def create_hist(self, bins):
        x = []
        for i in range(len(bins) - 1):
            x.append((bins[i] + bins[i + 1]) / 2)
        mean, std = self.predict(np.reshape(x, (-1, 1)), return_std=True) 

        h = ROOT.TH1F('h_gpr', 'GPR', len(bins) - 1, bins)
        for i in range(len(x)):
            h.SetBinContent(i + 1, mean[i])
            h.SetBinError(i + 1, std[i])

        return h


class ContourScanner:
    '''
    This class does the full marginal likelihood contour scan to arrive at the marginal
    posterior. Simply initialize then call [scan].

    Results of the scan are stored as a myriad of member variables, listed in [__init__].
    '''
    def __init__(self, fitter, sr_def, constant_factor, length_scale, integral_weights=None):
        '''
        @param fitter
            A [GPR] object defined above.
        @param sr_def
            The (min, max) of the sr region definition.
        @param constant_factor, length_scale
            The log-spaced list of values for the gamma variance factor and the length
            scale hyperparameters. These should be values as-is (not pre-logged).
        '''
        self.fitter : GPR = fitter # Make sure to pass in [fitter] after fitting already!
        self.sr_def = sr_def
        self.constant_factor = constant_factor
        self.length_scale = length_scale
        self.integral_weights = integral_weights

        ### Scikit optimized theta ###
        self.scikit_theta = np.exp(fitter.gpr.kernel_.theta)
        self.scikit_nlml = -self.fitter.gpr.log_marginal_likelihood()
        self.scikit_int = self.fitter.integral(self.sr_def)
        self._set_optimal_members('scikit')

        ### Marginal likelihood thresholds ###
        # Computing the integral is slow, so only do it for the 1 or 2 sigma CI
        self.max_nlml_1sigma = self.scikit_nlml + get_chi2_quantile(1, 2) / 2
        self.max_nlml_2sigma = self.scikit_nlml + get_chi2_quantile(2, 2) / 2
        self.max_nlml_for_yields = self.max_nlml_1sigma

        # Used for setting the z axis scale nicely
        self.min_nlml = None 
        self.max_nlml = None

        # Get most diagonal points in the 1sigma CI for comparison
        self.min_scale_theta = None # (gamma factor, length scale)
        self.max_scale_theta = None
        self.min_scale_int = None # (int, err)
        self.max_scale_int = None 

        ### Contour histograms ###
        self.h_nlml   = ROOT.TH2F('h_nlml'  , '', len(length_scale) - 1, length_scale, len(constant_factor) - 1, constant_factor)
        self.h_int    = ROOT.TH2F('h_int'   , '', len(length_scale) - 1, length_scale, len(constant_factor) - 1, constant_factor)
        self.h_int_1s = ROOT.TH2F('h_int_1s', '', len(length_scale) - 1, length_scale, len(constant_factor) - 1, constant_factor)

        ### p(f) histograms ###
        self.h_pf = None
        self.h_pf_1s = None
        self.h_pf_weighted = None
        

    def scan(self):
        # These are centered on the x axis around self.scikit_int. 
        self.h_pf = ROOT.TH1D('h_pf', '', 200, 0.5 * self.scikit_int[0], 1.5 * self.scikit_int[0]) # probability of each f value
        self.h_pf_1s = self.h_pf.Clone() # only scan 1 sigma range
        if self.integral_weights is not None:
            int_weighted, int_weighted_err = self.fitter.weighted_integral(self.sr_def, self.integral_weights)
            self.h_pf_weighted = ROOT.TH1D('h_pf_weighted', '', 200, 0.5 * int_weighted, 1.5 * int_weighted)

        # Get probability integrals for use in converting p(y|theta) to p(theta|y)
        # Assume constant prior p(theta) in log theta space
        sum_ml = 0 # total in grid (assume negligible outside) ~= p(y|X)
        sum_ml_yields = 0 # only when calculating yield (see max_nlml_for_yields)
        sum_ml_1s = 0 # only in the 1 sigma likelihood ratio range. This should be about 68% of sum_ml if the chi2 approx is accurate. 
        sum_ml_2s = 0 # only in the 2 sigma likelihood ratio range. This should be about 95% of sum_ml if the chi2 approx is accurate. 

        for y,factor in enumerate(self.constant_factor[:-1]):
            for x,scale in enumerate(self.length_scale[:-1]):
                self.fitter.refit([factor, scale])
                nlml = -self.fitter.gpr.log_marginal_likelihood() 
                ml = np.exp(-nlml)

                self.h_nlml.SetBinContent(x+1, y+1, nlml)
                if self.min_nlml is None or nlml < self.min_nlml: 
                    self.min_nlml = nlml
                    self.grid_opt_theta = [factor, scale]
                if self.max_nlml is None or nlml > self.max_nlml: self.max_nlml = nlml
                sum_ml += ml

                if nlml < self.max_nlml_2sigma:
                    sum_ml_2s += ml
                
                if nlml < self.max_nlml_1sigma:
                    sum_ml_1s += ml

                if nlml < max(self.max_nlml_1sigma, self.max_nlml_for_yields):
                    sum_ml_yields += ml
                    int_fit, int_err = self.fitter.integral(self.sr_def, test_points=6)
                    print('Integral', x, y, int_fit, end='\r')
                    self.h_int.SetBinContent(x+1, y+1, int_fit)

                    ### Accumulate the p(f | y,X) integral ###
                    for i in range(1, self.h_pf.GetNbinsX() + 1):
                        pf_given_theta = stats.norm.pdf(self.h_pf.GetBinCenter(i), loc=int_fit, scale=int_err)
                        self.h_pf.SetBinContent(i, self.h_pf.GetBinContent(i) + pf_given_theta * ml) # divide by p(y|X) and width at end
                        if nlml < self.max_nlml_1sigma:
                            self.h_pf_1s.SetBinContent(i, self.h_pf_1s.GetBinContent(i) + pf_given_theta * ml) # divide by p(y|X) and width at end
                    
                    ### Accumulate the weighted p(f | y,X) integral ###
                    if self.integral_weights is not None:
                        int_weighted, int_weighted_err = self.fitter.weighted_integral(self.sr_def, self.integral_weights)
                        for i in range(1, self.h_pf_weighted.GetNbinsX() + 1):
                            self.h_pf_weighted[i] += ml * stats.norm.pdf(self.h_pf_weighted.GetBinCenter(i), loc=int_weighted, scale=int_weighted_err)

                    if nlml < self.max_nlml_1sigma:
                        self.h_int_1s.SetBinContent(x+1, y+1, int_fit)

                        ### Get the min/max points of the 1 sigma region ###
                        # Generally small length scales will have small variance and vice versa,
                        # so can just look for the most diagonal points
                        if self.min_scale_theta is None or scale < self.min_scale_theta[1] or (scale == self.min_scale_theta[1] and factor < self.min_scale_theta[0]):
                            self.min_scale_theta = (factor, scale)
                            self.min_scale_int = (int_fit, int_err)
                        if self.max_scale_theta is None or scale > self.max_scale_theta[1] or (scale == self.max_scale_theta[1] and factor > self.max_scale_theta[0]):
                            self.max_scale_theta = (factor, scale)
                            self.max_scale_int = (int_fit, int_err)

        ### Optimal point ###
        self.fitter.gpr.theta_ = self.scikit_theta # reset to original to avoid side-effects
        _diff_nlml = abs(self.scikit_nlml - self.min_nlml)
        _diff_length_scale = abs(self.scikit_theta[1] - self.grid_opt_theta[1]) / self.scikit_theta[1]
        if _diff_nlml > 1 or _diff_length_scale > 0.1:
            print('\n\n')
            plot.warning('Scikit optimal point is different from grid search')
            print(f'    Scikit NLML: {self.scikit_nlml:6.2f}        Min NLML: {self.min_nlml:6.2f}')
            print(f'    Scikit ell:  {self.scikit_theta[1]:6.2f}        Min ell:  {self.grid_opt_theta[1]:6.2f}')
            print(f'    Scikit var:  {self.scikit_theta[0]:6.2f}        Min var:  {self.grid_opt_theta[0]:6.2f}')
            print('')
        
        ### Finalize h_pf histograms ###
        # print('1sigma CI fraction of grid', sum_ml_1s / sum_ml) # should be approx 68%
        # print('2sigma CI fraction of grid', sum_ml_2s / sum_ml) # should be approx 95%
        self.h_pf.Scale(self.h_pf.GetBinWidth(1) / sum_ml_yields) # this makes h_pf integrate to about 1
        self.h_pf_1s.Scale(self.h_pf_1s.GetBinWidth(1) / sum_ml_1s) # this makes h_pf_1s integrate to about 1
        self.h_pf_weighted.Scale(self.h_pf_weighted.GetBinWidth(1) / sum_ml_yields) # this makes it integrate to about 1
    
    def _set_optimal_members(self, name):
        self.optimal_theta = getattr(self, name + '_theta')
        self.optimal_nlml = getattr(self, name + '_nlml')
        self.optimal_int = getattr(self, name + '_int')
        self.fitter.gpr.theta_ = self.optimal_theta
        self.optimal_title = f'{self.fitter}'
        self.optimal_marker = ROOT.TMarker(self.optimal_theta[1], self.optimal_theta[0], ROOT.kFullCircle)
        self.optimal_marker.SetMarkerSize(2)


def gpr_likelihood_contours(
        config : FitConfig,
        h_cr : ROOT.TH1F, 
        h_data_sr: ROOT.TH1F = None,
        h_mc : ROOT.TH1F = None, 
        h_diboson : ROOT.TH1F = None, 
        filebase : str = '', 
        subtitle : list[str] = [], 
        vary_bin : tuple[float, float] = None, 
    ):
    '''
    Run function for the contour scan. This class does the scan and also handles plots and
    saving the results.

    Makes the following plots:
        1. likelihood contours
        2. SR integral error colz in the +1 sigma region
        3. event plot with means of best and +-1 sigma hyperparameters
        4. event plots for each individual GPR in (3), with error band
        5. p(f | X, y) distribution
    Assumes the GPR has two hyperparameters: length scale and overall variance (constant
    kernel).

    @param vary_bin
        A tuple (min, max) of the cross variable [var] that we're currently fitting. This
        used only with [fit_results].  
    @param filebase
        Base path and name for output plot files. Generated plots will postpend their
        names to this.
    @param h_mc
        If [h_cr] is derived from MC, pass this histogram with the original MC errors and
        includes the SR region, which will be used for the closure test.
    @param h_data_sr
        The SR data, which is used to obtain the predicted signal strength. If None, will
        use asimov data = diboson + gpr MMLE.

        WARNING in this case changing mu_diboson does nothing because [h_diboson] is the
        nominal unscaled sample.
    @param h_diboson
        The diboson MC sample. This is used to determine the diboson signal strength.
    '''
    ### MC ###
    if h_mc:
        mc_sr_yield = plot.integral_user(h_mc, config.sr_window, use_width=True, return_error=True)
    else:
        mc_sr_yield = None

    ### Fit ###
    fit_range = config.get_fit_range(vary_bin)
    fitter = GPR(config.gpr_version)
    fitter.fit(h_cr, fit_range)

    ### Diboson signal strength precalcs ###
    if h_diboson:
        weights, n_integral, h_sr = calculate_signal_strength_weights(
            fitter=fitter,
            h_sr=h_data_sr, 
            h_diboson=h_diboson, 
            sr_window=config.sr_window,
        )
    else:
        weights = None
    
    ### Scan theta ###
    constant_factor = np.logspace(*(np.log10(x) for x in fitter.noise_level_bounds), num=25)
    length_scale = np.logspace(*(np.log10(x) for x in fitter.length_scale_bounds), num=25)
    scanner = ContourScanner(fitter, config.sr_window, constant_factor, length_scale, integral_weights=weights)
    scanner.scan()

    ### Plot contours ###
    plot_opts = {
        'filename': filebase + 'cont_nlml',
        'subtitle': subtitle + [scanner.optimal_title],
        'z_range': [scanner.min_nlml - 1, min(scanner.max_nlml, scanner.min_nlml + 100)],
    }
    plot_nlml_contours(scanner.h_nlml, scanner.min_nlml, scanner.optimal_marker, **plot_opts)

    ### Plot integral yields ###
    plot_opts = {
        'filename': filebase + 'cont_yields',
        'subtitle': [
            *subtitle,
            scanner.optimal_title,
            f'Min nlml yield: {format_error(*scanner.optimal_int)}',
        ],
    }
    plot_yield_scan(scanner.h_int, scanner.optimal_marker, **plot_opts)

    ### Plot +-1 sigma lines ###
    plot_updown_fits(
        scanner=scanner, 
        h_cr=h_cr,
        h_mc=h_mc,
        sr_window=config.sr_window,
        filename=filebase + '{stub}',
        subtitle=subtitle,
        xtitle=f'm(J) [GeV]',
        ytitle='Events / Bin Width',
        x_range=[50, 250],
    )

    ### p(f|X,y) distribution ###
    pfs = plot_pf(scanner.h_pf, 
        filename=filebase + 'p_intr',
        subtitle=subtitle,
        mc_yield=mc_sr_yield,
    )

    ### Weighted integral p(f|X,y) distribution ###
    if h_diboson:
        pfs_weighted = plot_pf(scanner.h_pf_weighted, 
            filename=filebase + 'pf_weighted',
            subtitle=subtitle,
            xtitle='f #equiv Weighted SR Integral',
            mc_yield=weighted_integral(h_mc, *config.sr_window, weights) if h_mc else None,
        )

    ### CSV summary output ###
    if config.fit_results:
        w = vary_bin[1] - vary_bin[0]
        csv_base_args = {
            'lep': config.lepton_channel,
            'vary': config.var.name,
            'bin': f'{vary_bin[0]},{vary_bin[1]}',
            'variation': config.variation,
        }

        ### MC yield ###
        if h_mc:
            config.fit_results.set_entry(
                **csv_base_args,
                fitter='vjets_mc',
                val=mc_sr_yield[0] / w, 
                err_up=mc_sr_yield[1] / w, 
                err_down=mc_sr_yield[1] / w,
            )
        
        ### NLML fit ###
        config.fit_results.set_entry(
            **csv_base_args,
            fitter=config.gpr_version + '_nlml',
            val=scanner.optimal_int[0] / w, 
            err_up=scanner.optimal_int[1] / w, 
            err_down=scanner.optimal_int[1] / w,
        )

        ### Marginal posterior ###
        pf_val = pfs[0]
        pf_err_down = pf_val - pfs[1]
        pf_err_up = pfs[2] - pf_val
        config.fit_results.set_entry(
            **csv_base_args,
            fitter=config.gpr_version + '_marg_post',
            val=pf_val / w, 
            err_up=pf_err_up / w, 
            err_down=pf_err_down / w,
        )
        
        ### Diboson signal strength ###
        if h_diboson:
            ### Weighted average ###
            val = n_integral[0] - pfs_weighted[0]
            err_down = n_integral[1]**2 + (pfs_weighted[0] - pfs_weighted[1])**2
            err_up = n_integral[1]**2 + (pfs_weighted[2] - pfs_weighted[0])**2
            config.fit_results.set_entry(
                **csv_base_args,
                fitter=config.gpr_version + '_mu_weighted_average',
                val=val, 
                err_up=err_up**0.5, 
                err_down=err_down**0.5,
            )

            ### Integral - integral ###
            data_int = plot.integral_user(h_sr, config.sr_window, use_width=True, return_error=True)
            diboson_int = plot.integral_user(h_diboson, config.sr_window, use_width=True, return_error=True)
            
            num = data_int[0] - pf_val
            val = num / diboson_int[0]
            err_down = val * ((data_int[1]**2 + pf_err_down**2) / num**2 + diboson_int[1]**2 / diboson_int[0]**2)**0.5
            err_up = val * ((data_int[1]**2 + pf_err_up**2) / num**2 + diboson_int[1]**2 / diboson_int[0]**2)**0.5

            config.fit_results.set_entry(
                **csv_base_args,
                fitter=config.gpr_version + '_mu_integral',
                val=val, 
                err_up=err_up, 
                err_down=err_down,
            )

            scanner.mu_integral = (val, (err_up + err_down) / 2)

    return scanner
            

##########################################################################################
###                                       CONFIG                                       ###
##########################################################################################

class FitConfig:
    '''
    Main configuration class for a GPR fit against a single discriminating variable, like
    m(VV). One [FitConfig] instance represents a series of fits for each bin in the
    variable.

    @property var
        The cross variable being binned against. I.e. m(VV) or pT(J).
    @property variation
        The current variation being run. Normally this is the stub for the histogram name,
        with a few special cases:
            - 'nominal'
            - 'mu-dibosonX': Sets the diboson signal strength to X, where X is a float
              string. When [use_vjets_mc] is True, this means a contamination of (1 - X) *
              h_diboson is added to h_vjets. When it is False, this corresponds to the
              signal strength used to subtract h_diboson from h_cr. 
            - 'mu-ttbar*': Assumes [mu_ttbar] has been set using an up/down variation, but
              otherwise uses the same histograms.
            - 'mu-stop*': Assumes [mu_stop] has been set using an up/down variation, but
              otherwise uses the same histograms.
    @property use_vjets_mc
        Does the closure test using the V+jets MC as the data, setting errors as sqrt(N).
    @property mu_stop, mu_ttbar, mu_diboson
        Relevant signal strengths used in the data subtraction scheme. For mu_diboson, 
        also the amount of signal contamination (see [variation]).
    @property fit_results
        A common [FitResults] object that can be referenced.
    @property bins_x, bins_y
        The bin edges used for the fit. [bins_x] is for m(J), while [bins_y] is for [var].
        Note the number of bins given by [bins_y] corresponds to the number of fits
        actually done.
    @property output_dir
        Directory that output plots and files are saved to.
    @property gpr_version
        Kernel specification for [GPR].
    @property sr_window
        The signal region min,max definition in m(J). These should align with [bins_x].
    '''

    def __init__(
            self, 
            lepton_channel : int,
            var : utils.Variable,
            variation : str = 'nominal',
            use_vjets_mc : bool = False,
            output_plots_dir : str = './output',
            output_hists_dir : str = None,
            gpr_version : str = 'rbf',
            sr_window : tuple[float, float] = (72, 102),
            mu_stop : float = 1,
            mu_ttbar : float = 1,
        ):
        self.lepton_channel = lepton_channel
        self.var = var
        self.variation = variation
        self.use_vjets_mc = use_vjets_mc
        self.output_plots_dir = output_plots_dir
        self.output_hists_dir = output_hists_dir or output_plots_dir
        self.gpr_version = gpr_version
        self.sr_window = sr_window
        self.mu_stop = mu_stop
        self.mu_ttbar = mu_ttbar

        ### Outputs ###
        if not os.path.exists(self.output_plots_dir):
            os.makedirs(self.output_plots_dir)
        if not os.path.exists(self.output_hists_dir):
            os.makedirs(self.output_hists_dir)
        self.fit_results = FitResults(f'{self.output_hists_dir}/gpr_fit_results.csv') # original output_dir
        self.output_root_file_path = f'{self.output_hists_dir}/gpr_{self.lepton_channel}lep_vjets_yield.root'
        
        ### Binning ###
        self.bins_y = self.get_bins_y()

        ### Signal contamination ###
        if variation.startswith('mu-diboson'):
            self.mu_diboson = float(variation.removeprefix('mu-diboson'))
        else:
            self.mu_diboson = 1

    def get_bins_x(self, bin_y):
        out = None
        if self.var.name == "vv_m":
            if self.lepton_channel == 1:
                if bin_y[0] >= 2000:
                    out = [50, 72, 102, 150, 200, 250]
                elif bin_y[0] > 1000:
                    out = [50, 55, 60, 66, 72, 82, 92, 102, 125, 150, 175, 200, 250]
            elif self.lepton_channel == 2:
                if bin_y[0] >= 1400:
                    out = [50, 72, 102, 150, 200, 250]
                elif bin_y[0] > 1000:
                    out = [50, 55, 60, 66, 72, 82, 92, 102, 125, 150, 175, 200, 250]
                else:
                    out = [50, 55, 60, 66, 72] + list(np.arange(82, 250, 10))
        if out is None:
            out = np.concatenate(([50, 53, 56, 59], np.arange(62, 250, 5)))
        return np.array(out, dtype=float)

    def get_bins_y(self):
        return utils.get_bins(lepton_channel=self.lepton_channel, var=self.var)

    def get_fit_range(self, bin_y):
        # if self.var.name == "vv_m" and bin_y[0] >= 2000:
        #     return (50, 250)
        return (50, 250)


##########################################################################################
###                                       RUNNER                                       ###
##########################################################################################

def summary_actions_from_csv(config : FitConfig):
    '''
    Some final actions after running the full fit for a single variable. This reads
    data from the CSV file, so it can be run without doing the fits all over again
    using the --fromCsvOnly option.
    '''
    csv_base_spec = {
        'lep': config.lepton_channel, 
        'variation': config.variation,
        'vary': config.var.name, 
        'bins': config.bins_y,
    }

    ### Plot summary distribution ###
    fitters = [
        'vjets_mc',
        f'{config.gpr_version}_nlml',
        f'{config.gpr_version}_marg_post',
    ]
    legend = [
        'MC',
        'GPR MMLE Fit',
        'GPR Marg Post',
    ]
    if not config.use_vjets_mc:
        fitters = fitters[1:]
        legend = legend[1:]

    graphs = [config.fit_results.get_graph(**csv_base_spec, fitter=x, unscale_width=True) for x in fitters]
    plot_summary_distribution(
        graphs,
        filename=f'{config.output_plots_dir}/gpr_{config.lepton_channel}lep_{config.var}_summary',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
        ],
        legend=legend,
        xtitle=f'{config.var:title}',
        edge_labels=[str(x) for x in config.bins_y],
        subplot2='ratios' if config.use_vjets_mc else 'errors',
        subplot3='errors' if config.use_vjets_mc else None,
    )

    ### Save output histogram ###
    f_output = ROOT.TFile(config.output_root_file_path, 'UPDATE')
    
    # This naming convention is pretty important to match that in the CxAODReader outputs
    # Since ResonanceFinder uses the same convention for ALL samples.
    if config.variation == utils.variation_nom:
        histname = f'Vjets_SR_{config.var}'
    else:
        histname = f'Vjets_SR_{config.var}__{config.variation}'
    h = config.fit_results.get_histogram(
        **csv_base_spec, 
        fitter=config.gpr_version + '_marg_post', 
        histname=histname,
    )
    h.Write()


def run(
        file_manager : utils.FileManager,
        config : FitConfig,
        from_csv_only : bool = False,
        skip_if_in_csv : bool = False,
    ):
    '''
    This runs all the bin fits using the full contour scan (marginal posterior) for a
    single [config] (i.e. channel, variable, and variation).
    '''
    ### Short circuit ###
    if from_csv_only:
        summary_actions_from_csv(config)
        return
    
    ### Retrieve histograms ###
    hist_name = f'{{sample}}_VV{{lep}}_Merg_{config.var}__v__fatjet_m'
    h_diboson = file_manager.get_hist(config.lepton_channel, utils.Sample.diboson, hist_name, config.variation)
    if config.use_vjets_mc:
        h_wjets = file_manager.get_hist(config.lepton_channel, utils.Sample.wjets, hist_name, config.variation)
        h_zjets = file_manager.get_hist(config.lepton_channel, utils.Sample.zjets, hist_name, config.variation)

        h_vjets = h_wjets.Clone()
        h_vjets.Add(h_zjets)
        h_vjets_diboson = h_vjets.Clone() # vjets + diboson, for signal strength estimate
        if config.mu_diboson != 1:
            h_vjets.Add(h_diboson, 1 - config.mu_diboson) 
        
        h_data = None
    else:
        h_data  = file_manager.get_hist(config.lepton_channel, utils.Sample.data,  hist_name, config.variation)
        h_ttbar = file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, hist_name, config.variation)
        h_stop  = file_manager.get_hist(config.lepton_channel, utils.Sample.stop,  hist_name, config.variation)

        h_ttbar = h_ttbar.Clone() # This clone is really important! Or else the next call will be double scaled, etc.
        h_stop = h_stop.Clone() # This clone is really important! Or else the next call will be double scaled, etc.
        h_ttbar.Scale(config.mu_ttbar)
        h_stop.Scale(config.mu_stop)
        
        h_vjets = h_data.Clone()
        h_vjets.Add(h_ttbar, -1)
        h_vjets.Add(h_stop, -1)
        h_vjets_diboson = h_vjets.Clone() # vjets + diboson, for signal strength estimate
        h_vjets.Add(h_diboson, -config.mu_diboson) 

    ### Run for each bin ###
    for i in range(len(config.bins_y) - 1):
        ### Common options ###
        bin_y = (config.bins_y[i], config.bins_y[i+1])
        binstr = f'{bin_y[0]},{bin_y[1]}'
        common_subtitle = [
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.var.title} #in {bin_y} {config.var.unit}',
        ]

        if skip_if_in_csv and config.fit_results.contains(config.lepton_channel, config.var.name, config.variation, f'{config.gpr_version}_marg_post', binstr):
            plot.notice(f'gpr.py::run() Skipping {config.lepton_channel}lep ({config.variation}) {config.var}=[{binstr}]')
            continue

        ### Histogram manipulation ###
        bins_x = config.get_bins_x(bin_y)
        def prepare_bin(h, bins=bins_x):
            h = plot.projectX(h, bin_y)
            h = plot.rebin(h, bins)
            h.Scale(1, 'width')
            return h
        
        h_vjets_bin = prepare_bin(h_vjets)
        h_diboson_bin = prepare_bin(h_diboson) if h_diboson else None
        h_vjets_diboson_bin = prepare_bin(h_vjets_diboson) if h_vjets_diboson else None
        
        if config.use_vjets_mc:
            h_mc = h_vjets_bin.Clone()
            set_sqrtn_errors(h_vjets_bin, width_scaled=True)
        else:
            h_mc = None
        _, h_cr = get_sr_cr(h_vjets_bin, config.sr_window)

        ### Run fit ###
        plot.notice(f'gpr.py::run() Running {config.lepton_channel}lep ({config.variation}) {config.var}=[{binstr}]')
        contour_scanner = gpr_likelihood_contours(
            h_cr=h_cr, 
            h_data_sr=h_vjets_diboson_bin, 
            h_mc=h_mc,
            h_diboson=h_diboson_bin,
            vary_bin=bin_y,
            filebase=f'{config.output_plots_dir}/gpr_{config.lepton_channel}lep_{config.var}_{config.variation}_{binstr}_',
            config=config,
            subtitle=common_subtitle,
        )

        ### Postfit plot ###
        if h_data:
            contour_scanner.fitter.refit(contour_scanner.scikit_theta)
            fit_range = config.get_fit_range(bin_y)
            i = 0
            for x in bins_x:
                if x > fit_range[1]:
                    break
                i += 1
            bins = bins_x[:i]
            h_gpr = contour_scanner.fitter.create_hist(bins)
            plot_postfit(
                h_gpr=h_gpr,
                h_data=prepare_bin(h_data, bins),
                h_ttbar=prepare_bin(h_ttbar, bins),
                h_stop=prepare_bin(h_stop, bins),
                h_diboson=prepare_bin(h_diboson, bins),
                filename=f'{config.output_plots_dir}/gpr_{config.lepton_channel}lep_{config.var}_{config.variation}_{binstr}_postfit',
                subtitle=common_subtitle,
                sr_window=config.sr_window,
                mu_diboson=contour_scanner.mu_integral[0],
            )
        contour_scanner = None # this cleans up the ROOT objects held in memory by contour_scanner

    summary_actions_from_csv(config)


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################


def parse_args():
    parser = ArgumentParser(
        description="", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument("--lepton", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--var", required=True, help='Variable to fit against; this must be present in a histogram {var}__v__fatjet_m and in the variable.py module.')
    
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('--output-hists', default=None, help='Alternative directory to output the main CSV/histograms to. Defaults to same directory as [--output].')
    parser.add_argument('--closure-test', action='store_true', help='Set this flag to fit and do a closure test against the V+jets MC.')
    parser.add_argument('--mu-ttbar', default=1, type=float, help='Scale factor for the ttbar sample. Default = 1.')
    parser.add_argument('--mu-stop', default=1, type=float, help='Scale factor for the stop sample. Default = 1.')
    parser.add_argument('--from-csv-only', action='store_true', help="Don't do the fit, just save the fit results in the CSV to a ROOT histogram.")
    parser.add_argument('--variation', default='nominal')
    return parser.parse_args()


def get_files(filepaths):
    file_manager = utils.FileManager(
        samples=[
            utils.Sample.wjets,
            utils.Sample.zjets,
            utils.Sample.ttbar,
            utils.Sample.stop,
            utils.Sample.diboson,
            utils.Sample.data,
        ],
        file_path_formats=filepaths,
    )
    return file_manager


def main():
    '''
    See file header.
    '''
    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    args = parse_args()
    file_manager = get_files(args.filepaths)
    config = FitConfig(
        lepton_channel=args.lepton,
        var=getattr(utils.Variable, args.var),
        variation=args.variation,
        use_vjets_mc=args.closure_test,
        output_plots_dir=args.output,
        output_hists_dir=args.output_hists,
        mu_stop=args.mu_stop,
        mu_ttbar=args.mu_ttbar,
    )
    run(file_manager, config, args.from_csv_only)

    
if __name__ == '__main__':
    main()
