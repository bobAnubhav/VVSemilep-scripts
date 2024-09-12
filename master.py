#!/usr/bin/env python3
'''
@file master.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 20, 2024
@brief Master run script for doing the unfolded analysis for VVsemilep

------------------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------------------

    setupATLAS
    lsetup "root recommended"
    lsetup "python centos7-3.9"

Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(. Alternatively, you can just setup ResonanceFinder.

------------------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------------------

Check [utils.Sample] and [utils.Variable] to make sure the hardcoded naming stuctures are
correct. Check [utils.get_bins] and [gpr.FitConfig.get_bins_x] to ensure the desired binnings
are set.

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    main.py filepath/formatter_1.root [...]

This will fetch histogram files using the naming convention supplied in the arguments.
These arguments can include python formatters (using curly braces) for 'lep', which will
be replaced with the lepton channel name, and 'sample', which uses
[utils.Sample.file_stubs]. For example,

    hists/{lep}/{sample}.root

See [utils.FileManager] for details.
'''

from __future__ import annotations

import ROOT # type: ignore

import numpy as np
import os
import sys
import re
import gc
import subprocess
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from plotting import plot
import utils
import unfolding
import gpr
import ttbar_fit
import diboson_fit
import merge_gpr_condor

##########################################################################################
###                                        UTILS                                       ###
##########################################################################################

def convert_alpha(alpha : tuple[float, float], orig : tuple[float, float]) -> tuple[float, float]:
    '''
    For a Gaussian constrained floating parameter in the PLU fit, given a result w/ error
    in terms of [alpha], returns the actual values.
    '''
    val = orig[0] + alpha[0] * orig[1]
    err = alpha[1] * orig[1]
    return (val, err)


_condorSubmitHead = '''# Setup
Universe = vanilla
getenv = True

# Log paths
output = {output_dir}/condor_logs/$(ClusterId).$(ProcId).out
error = {output_dir}/condor_logs/$(ClusterId).$(ProcId).err
log = {output_dir}/condor_logs/$(ClusterId).log

# Queue
+JobFlavour = "espresso"
+queue="short"

# Retry failed jobs
on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
max_retries = 3

# Command
'''

class CondorSubmitMaker:
    '''
    Base class for creating and submitting HTCondor batch jobs. Derived classes should
    implement [write_executable] as well as some function to add the arguments.

    Usage:
        maker = CondorSubmitMaker(config, 'submit.condor')
        ...[Add arguments with derived class]...
        maker.close() # finishes writing the submit file
        maker.run() # submits the jobs
        maker.wait() # awaits completion
    '''
    def __init__(self, config : GlobalConfig, filepath : str):
        self.config = config
        self.filepath = filepath
        if not os.path.exists(f'{config.output_dir}/condor_logs'):
            os.makedirs(f'{config.output_dir}/condor_logs')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        self.f = open(filepath, 'w')
        self.f.write(_condorSubmitHead.format(output_dir=config.output_dir))
        self.write_executable()

        self.f.writelines([
            '\n',
            '# Queue\n',
            'queue arguments from (\n',
        ])

    def write_executable(self):
        raise NotImplementedError('CondorSubmitMaker() Derived classes must implement write_executable()')

    def close(self):
        self.f.write(')')
        self.f.close()
        plot.success(f'Wrote condor submit file to {self.filepath}')
    
    def run(self):
        self.res = subprocess.run(['condor_submit', self.filepath], capture_output=True, text=True)
        if self.res.returncode == 0:
            search_string = '(\d*) job\(s\) submitted to cluster (\d*).'
            m = re.search(search_string, self.res.stdout)
            if m is not None and len(m.groups()) == 2:
                self.n_jobs = int(m.groups()[0])
                self.cluster = m.groups()[1]
                plot.success(f"Submitted {self.n_jobs} jobs to cluster {self.cluster}")
                return self.cluster
            else:
                plot.error(f"Couldn't parse condor_submit output:\n{self.res.stdout}")
                return None
        else:
            plot.error(f"Couldn't launch condor jobs:\n{self.res}")
            return None
        
    def wait(self):
        done = False
        while not done:
            subprocess.run(['condor_q', str(self.cluster)])
            res = subprocess.run(['condor_wait', f'{self.config.output_dir}/condor_logs/{self.cluster}.log', '-wait', '10'], capture_output=True, text=True)
            done = res.returncode == 0
        
        ### Check jobs completed without errors ###
        res = subprocess.run(['condor_history', str(self.cluster), '-af', 'clusterId', 'procId', 'ExitStatus'], capture_output=True, text=True)
        res_lines = res.stdout.splitlines()
        bad_list = []
        for x in res_lines:
            cluster, proc, status = x.split(' ')
            if status != '0':
                bad_list.append(proc)
        if bad_list:
            plot.error(f"{len(bad_list)} jobs failed in cluster {self.cluster}")
            raise RuntimeError(f"{len(bad_list)} jobs failed in cluster {self.cluster}")
        if len(res_lines) != self.n_jobs:
            plot.warning(f"condor_history returned a different number of jobs ({len(res_lines)}) from expected ({self.n_jobs})")

        plot.success(f"All {self.n_jobs} jobs from cluster {self.cluster} have completed")


class GprSubmitMaker(CondorSubmitMaker):
    '''
    Creates the condor.submit file for launching GPR jobs on HT Condor batch systems.

    Usage:
        maker = GprSubmitMaker(config, 'submit.condor')
        maker.add(gpr_config1)
        maker.add(gpr_config2)
        ...
        maker.close()
    '''
    def __init__(self, config: GlobalConfig, filepath: str):
        super().__init__(config, filepath)

    def write_executable(self):
        self.f.writelines([
            "Executable = gpr.py\n",
            "transfer_input_files = plotting,utils.py\n",
        ])
    
    def add(self, fit_config : gpr.FitConfig):
        input_paths = [os.path.abspath(x) for x in self.config.file_manager.file_path_formats]
        output_path = os.path.abspath(fit_config.output_plots_dir)
        # Output hists and plots to same finely binned directory so no overwrite between jobs!

        self.f.write('    ')
        self.f.write(' '.join(input_paths))
        self.f.write(f' --lepton {fit_config.lepton_channel}')
        self.f.write(f' --var {fit_config.var}')
        self.f.write(f' --output {output_path}')
        if self.config.is_asimov:
            self.f.write(f' --closure-test')
        self.f.write(f' --variation {fit_config.variation}')
        self.f.write(f' --mu-ttbar {fit_config.mu_ttbar}')
        self.f.write(f' --mu-stop {fit_config.mu_stop}\n')


class RankingSubmitMaker(CondorSubmitMaker):
    '''
    Creates the condor.submit file for launching the ranking plot jobs.

    Usage:
        maker = RankingSubmitMaker(cfg, "submit.condor")
        maker.add(systs, ...)
    '''
    def __init__(self, config: GlobalConfig, filepath: str, ):
        super().__init__(config, filepath)

    def write_executable(self):
        self.f.writelines([
            "Executable = ranking.py\n",
            "transfer_input_files = ResonanceFinder/NPCheck\n",
            "request_memory = 512MB\n",
        ])
    
    def add(self, systs: list[str], ws_path: str, fcc_path: str, output_path: str, asimov: bool = False):
        output_path = os.path.abspath(output_path)
        os.makedirs(output_path, exist_ok=True)

        for syst in systs:
            self.f.write('    ')
            self.f.write(' '.join([
                ws_path,
                syst,
                fcc_path,
                'NONE', # limit path
                '1', # postfit
                str(int(asimov)), # asimov
                output_path,
                os.getcwd() + "/ResonanceFinder/NPCheck/macros/makeRanking.C" # macro path
            ]))
            self.f.write('\n')


##########################################################################################
###                                        PLOTS                                       ###
##########################################################################################

def _reassign_bins(h, bins):
    h_new = ROOT.TH1F(h.GetName() + '_adj', h.GetTitle(), len(bins) - 1, bins)
    for i in range(len(bins) + 2):
        h_new.SetBinContent(i, h.GetBinContent(i))
        h_new.SetBinError(i, h.GetBinError(i))
    return h_new


def plot_gpr_mu_diboson_correlations(
        config : gpr.FitConfig,
        yields : list[float],
        filename : str,
    ):
    '''
    This plots the change in yields as the diboson signal strength guess is varied. This
    function also returns the average delta_N/delta_mu for each bin in [config.bins_y].
    '''
    ### Get graphs ###
    csv_base_spec = {
        'lep': config.lepton_channel,
        'fitter': f'{config.gpr_version}_marg_post',
        'vary': config.var.name,
        'bins': config.bins_y,
    }
    i_nom = len(yields) // 2
    graphs = [config.fit_results.get_graph(**csv_base_spec, variation=f'mu-diboson{x}', unscale_width=True) for x in yields]
    graphs.insert(i_nom, config.fit_results.get_graph(**csv_base_spec, variation='nominal', unscale_width=True))
    legend = [str(x) for x in yields]
    legend.insert(i_nom, '1.0')

    ### Get ratios ###
    average_corr_factor = np.zeros(len(config.bins_y) - 1) # this is the correction factor that should be ADDED to the measured GPR yield (scaled by 1-mu)
    h_nom = graphs[i_nom]
    def subplot(hists, **kwargs):
        # Use the callback here to copy the formatting from [hists]
        ratios = []
        for h,mu in zip(hists, legend):
            mu = float(mu)
            if mu == 1: continue

            h = h.Clone()
            for i in range(h.GetN()):
                delta_delta = (h.GetPointY(i) - h_nom.GetPointY(i)) / h_nom.GetPointY(i) / (1 - mu)
                average_corr_factor[i] += (h.GetPointY(i) - h_nom.GetPointY(i)) / (1 - mu)
                h.SetPointY(i, delta_delta)
                h.SetPointEYhigh(i, 0)
                h.SetPointEYlow(i, 0)
            ratios.append(h)
        return ratios

    ### Plot ###
    gpr.plot_summary_distribution(
        hists=graphs,
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'Scanning #mu^{diboson}_{guess}',
        ],
        legend=legend,
        ytitle='Events',
        ytitle2='#frac{#DeltaFit / Fit_{nom}}{#Delta#mu}',
        xtitle=f'{config.var:title}',
        edge_labels=[str(x) for x in config.bins_y],
        subplot2=subplot,
        subplot3=None,
        opts2='P',
        y_range2=[0, 0.18],
        ydivs2=503,
    )

    return average_corr_factor / len(yields)


def plot_gpr_ttbar_and_stop_correlations(config : gpr.FitConfig, filename : str):
    '''
    This plots the change in yields as the ttbar and stop signal strengths are varied.
    '''
    ### Get graphs ###
    csv_base_spec = {
        'lep': config.lepton_channel,
        'fitter': f'{config.gpr_version}_marg_post',
        'vary': config.var.name,
        'bins': config.bins_y,
    }
    variations = ['nominal'] + [x + y for x in [utils.variation_mu_ttbar, utils.variation_mu_stop] for y in [utils.variation_up_key, utils.variation_down_key]]
    graphs = [config.fit_results.get_graph(**csv_base_spec, variation=x, unscale_width=True) for x in variations]

    gpr.plot_summary_distribution(
        hists=graphs,
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
        ],
        legend=variations,
        ytitle='Events',
        ytitle2='Var / Nom',
        xtitle=f'{config.var:title}',
        edge_labels=[str(x) for x in config.bins_y],
        subplot2='ratios',
        subplot3=None,
        y_range2=[0.9, 1.1],
        ydivs2=503,
    )


def plot_gpr_mc_comparisons(config: SingleChannelConfig, filename : str):
    ### Systematic adder ###
    def add_errs(err_cum, get_errs):
        '''
        Adds errors from [get_errs] to [err_cum] in quadrature.

        @param get_errs
            A function that takes a variation name and returns a set of (absolute) errors.
        '''
        err_cum = err_cum ** 2
        for vari in utils.variations_hist:
            diffs_up = get_errs(vari + utils.variation_up_key)
            diffs_down = get_errs(vari + utils.variation_down_key)
            errs = (diffs_up + diffs_down) / 2
            err_cum += errs ** 2
        return np.sqrt(err_cum)

    ### Get GPR nominal ###
    gpr_config = config.gpr_nominal_config
    csv_base_spec = {
        'lep': config.lepton_channel,
        'fitter': f'{gpr_config.gpr_version}_marg_post',
        'vary': gpr_config.var.name,
        'bins': gpr_config.bins_y,
    }
    g_gpr_nom = config.gpr_results.get_graph(**csv_base_spec, variation=utils.variation_nom, unscale_width=True)

    ### Get GPR with systematics ###
    def get_diffs(vari):
        g = config.gpr_results.get_graph(**csv_base_spec, variation=vari, unscale_width=True)
        return np.array([abs(g.GetPointY(i) - g_gpr_nom.GetPointY(i)) for i in range(g.GetN())])

    err_cum = np.array([g_gpr_nom.GetErrorY(i) for i in range(g_gpr_nom.GetN())])
    err_cum = add_errs(err_cum, get_diffs)

    g_gpr_systs = g_gpr_nom.Clone()
    for i in range(g_gpr_systs.GetN()):
        g_gpr_systs.SetPointEYhigh(i, err_cum[i])
        g_gpr_systs.SetPointEYlow(i, err_cum[i])

    ### Get MC nominal ###
    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + config.var_reader_name
    def get_hist(variation):
        h_wjets = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.wjets, hist_name, variation=variation)
        h_zjets = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.zjets, hist_name, variation=variation)
        h_vjets = h_wjets.Clone()
        h_vjets.Add(h_zjets)
        return plot.rebin(h_vjets, gpr_config.bins_y)
    
    h_mc_nom = get_hist(utils.variation_nom)
    
    ### Yi's MadGraph comparison ###
    def get_madgraph_shape_error():
        '''
        Gets an uncertainty by taking the difference between Sherpa and Madgraph, using
        Yi's smoothed histograms stored in `hists/vjets_theory_uncert_smooth.root`.
        
        Note these are ratio plots from Sherpa, with values around 1.0. So first scale
        back by the Sherpa. Then need to rebin, but the bins don't align. So do a linear
        split along nearest bins.
        '''
        temp_file_manager = utils.FileManager(
            samples=[utils.Sample.wjets, utils.Sample.zjets],
            file_path_formats=['hists/vjets_theory_uncert_smooth.root'],
            lepton_channels=[config.lepton_channel],
        )
        h_wjets_mg = temp_file_manager.get_hist(config.lepton_channel, utils.Sample.wjets, hist_name.format(sample='Wjets', lep='{lep}'))
        h_zjets_mg = temp_file_manager.get_hist(config.lepton_channel, utils.Sample.zjets, hist_name.format(sample='Zjets', lep='{lep}'))
        
        ### Scale by Sherpa ###
        h_wjets_sherpa = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.wjets, hist_name)
        h_zjets_sherpa = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.zjets, hist_name)
        
        bins_old = np.array([h_wjets_mg.GetBinLowEdge(i) for i in range(1, len(h_wjets_mg))])
        h_wjets_sherpa = plot.rebin(h_wjets_sherpa, bins_old)
        h_zjets_sherpa = plot.rebin(h_zjets_sherpa, bins_old)

        h_mg_old = h_wjets_mg.Clone()
        for i in range(1, h_mg_old.GetNbinsX() + 1):
            h_mg_old[i] = h_wjets_mg[i] * h_wjets_sherpa[i] + h_zjets_mg[i] * h_zjets_sherpa[i]

        ### Rebin ###
        bins_new = np.array(gpr_config.bins_y, dtype=float)
        i_new = 0
        i_old = 0
        h_mg_new = ROOT.TH1F('h_mg_new', '', len(bins_new) - 1, bins_new)
        for i_new in range(len(bins_new) - 1):
            for i_old in range(len(bins_old) - 1):
                overlap_range = (max(bins_old[i_old], bins_new[i_new]), min(bins_old[i_old + 1], bins_new[i_new + 1]))
                if overlap_range[1] > overlap_range[0]: 
                    frac = (overlap_range[1] - overlap_range[0]) / (bins_old[i_old + 1] - bins_old[i_old])
                    h_mg_new[i_new + 1] += h_mg_old[i_old + 1] * frac

        ### Final diff with nominal sherpa ###
        return np.array([abs(h_mg_new[i] - h_mc_nom[i]) for i in range(1, h_mg_new.GetNbinsX() + 1)])

    ### Get MC with systematics ###
    def get_diffs(vari):
        h_var = get_hist(vari)
        return np.array([abs(h_var[i] - h_mc_nom[i]) for i in range(1, h_var.GetNbinsX() + 1)])

    err_cum = np.array([h_mc_nom.GetBinError(i) for i in range(1, h_mc_nom.GetNbinsX() + 1)])
    err_cum = add_errs(err_cum, get_diffs)
    try:
        err_mg = get_madgraph_shape_error()
        err_cum = np.sqrt(err_cum ** 2 + err_mg ** 2)
    except Exception as e:
        plot.warning('master.py::plot_gpr_mc_comparisons() Unable to get MadGraph shape error')

    h_mc_systs = h_mc_nom.Clone()
    for i in range(1, h_mc_systs.GetNbinsX() + 1):
        h_mc_systs.SetBinError(i, err_cum[i - 1])

    ### Plot ###
    gpr.plot_summary_distribution(
        hists=[h_mc_nom, h_mc_systs, g_gpr_nom, g_gpr_systs],
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel SR',
        ],
        legend=['MC Stat. Only', 'MC Syst.', 'GPR Nominal', 'GPR + Syst. Corr.'],
        ytitle='Events',
        ytitle2='#frac{GPR}{MC}',
        xtitle=f'{gpr_config.var:title}',
        edge_labels=[str(x) for x in gpr_config.bins_y],
        ydivs2=503,
        ratio_denom=lambda i: 0 if i >= 2 else None,
    )


def _plot_yield_comparison(filename, h_fit, h_mc, h_eft=None, eft_legend=None, **plot_opts):
    '''
    Plots a 1D comparison between a fit and MC. The fit is shown as data points while the
    MC is shown as a filled blue band. Includes a fit/MC ratio.
    '''
    h_mc.SetMarkerSize(0)
    h_mc.SetLineColor(plot.colors.blue)
    h_mc_err = h_mc.Clone()
    h_mc_err.SetFillColorAlpha(plot.colors.blue, 0.3)

    if h_eft:
        h_eft.SetMarkerSize(0)
        h_eft.SetLineColor(plot.colors.orange)
        h_eft_err = h_eft.Clone()
        h_eft_err.SetFillColorAlpha(plot.colors.orange, 0.3)

    ratio = h_fit.Clone()
    ratio.Divide(h_mc)

    if h_eft:
        objs = [h_mc, h_mc_err, h_eft, h_eft_err, h_fit]
        legend = ['', 'SM', '', f'SM + {eft_legend}', 'Fit']
        opts = ['HIST', 'E2', 'HIST', 'E2', 'PE']
        legend_opts = ['', 'FL', '', 'FL', 'PEL']
    else:
        objs = [h_mc, h_mc_err, h_fit]
        legend = ['', 'MC', 'Fit']
        opts = ['HIST', 'E2', 'PE']
        legend_opts = ['', 'FL', 'PEL']

    ROOT.gStyle.SetErrorX(0)
    for logy in [True, False]:
        plot.plot_ratio(
            filename = filename + '_logy' if logy else filename,
            objs1=objs,
            objs2=[ratio],
            legend=legend,
            opts=opts,
            legend_opts=legend_opts,
            opts2='PE0',
            ytitle='Events',
            ytitle2='Fit / SM' if h_eft else 'Fit / MC',
            hline=1,
            y_range2=(0.5, 1.5),
            logy=logy,
            **plot_opts,
        )
    ROOT.gStyle.SetErrorX()


def plot_naive_bin_yields(config : SingleChannelConfig, filename : str):
    '''
    Plots the bin-by-bin yields taken by just doing Data - Bkgs.
    '''
    ### Adjusted bins ###
    bins = config.bins #utils.get_adjusted_bins(config.lepton_channel, config.variable)

    ### Get hists ###
    f_gpr = ROOT.TFile(f'{config.gbl.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + config.variable.name)
    h_gpr = _reassign_bins(h_gpr, bins)

    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + config.var_reader_name
    def get_hist(sample):
        h = config.gbl.file_manager.get_hist(config.lepton_channel, sample, hist_name)
        h = plot.rebin(h, config.bins)
        return _reassign_bins(h, bins)
    
    h_diboson = get_hist(utils.Sample.diboson)
    h_ttbar = get_hist(utils.Sample.ttbar)
    h_stop =  get_hist(utils.Sample.stop)
    h_data =  get_hist(utils.Sample.data)

    h_yield = h_data.Clone()
    h_yield.Add(h_gpr, -1)
    h_yield.Add(h_ttbar, -config.gbl.ttbar_fitter.mu_ttbar_nom[0])
    h_yield.Add(h_stop, -config.gbl.mu_stop[0])

    _plot_yield_comparison(
        h_fit=h_yield,
        h_mc=h_diboson,
        filename=filename,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'Naive diboson yield (Data - Bkg)',
        ],
        xtitle=f'{config.variable:title}',
    )


def plot_plu_yields(config : SingleChannelConfig, plu_fit_results, filename : str):
    '''
    Uses [plot_yield_comparison] to plot the PLU unfolded result against the fiducial MC.
    '''
    ### Adjusted bins ###
    bins = config.bins #utils.get_adjusted_bins(config.lepton_channel, config.variable)

    ### Fit ###
    h_fit = ROOT.TH1F('h_fit', '', len(bins) - 1, bins)
    for i in range(1, len(bins)):
        mu = plu_fit_results[f'mu_{i:02}']
        h_fit.SetBinContent(i, mu[0])
        h_fit.SetBinError(i, mu[1])
    
    ### MC ###
    # Buggy response matrix in eos_v3 files, so point to local files in interim
    # Note fixed in https://gitlab.cern.ch/CxAODFramework/CxAODReader_VVSemileptonic/-/merge_requests/445
    if 'v3' in config.gbl.file_manager.file_path_formats[0]:
        plot.warning('master.py::plot_plu_yields() Not using v3 histograms with buggy response matrix. Using hardcoded local path!')
        temp_file_manager = utils.FileManager(
        samples=[utils.Sample.diboson],#, utils.Sample.cw_lin, utils.Sample.cw_quad],
        file_path_formats=['/eos/user/a/anubhav/phd/cxAOD_out_grid/hist_28July/1lep_diboson_hist.root'], # histogram with correct response matrx with correct fiducial weight 
        lepton_channels=[1]#[0, 1, 2],
        )
        # for cw lin and quad 
        temp_file_manager2 = utils.FileManager(
        samples=[utils.Sample.cw_lin, utils.Sample.cw_quad],
        file_path_formats= ['/afs/cern.ch/user/a/anubhav/private/Riley_VVsemilep/VVSemilep-scripts/ hists-May24/{lep}_{sample}_x_May24.hists.root'],
        lepton_channels=[0,1,2]
        )

    else:
        temp_file_manager = config.gbl.file_manager
    def get(sample):
        # h = config.gbl.file_manager.get_hist(config.lepton_channel, sample, '{sample}_VV{lep}_Merg_unfoldingMtx_' + variable.name)
        h = temp_file_manager.get_hist(config.lepton_channel, sample, '{sample}_VV{lep}_Merg_unfoldingMtx_' + config.variable.name)
        h = plot.rebin(h.ProjectionX(), config.bins)
        return _reassign_bins(h, bins)
    h_mc = get(utils.Sample.diboson)

    ### EFT ###
    # Changed file_manager to May-hist for lin and quad cw I dont have it yet 
    temp_file_manager = temp_file_manager2 
    cw = 0.12
    h_cw_quad = get(utils.Sample.cw_quad)
    h_cw_lin = get(utils.Sample.cw_lin)
    h_cw_quad.Scale(cw**2)
    h_cw_lin.Scale(cw)
    h_cw_quad.Add(h_cw_lin)
    h_cw_quad.Add(h_mc)

    ### Plot ###
    yield_args = dict(
        h_fit=h_fit,
        h_mc=h_mc,
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel fiducial region',
            'MC errors are stat only',
        ],
        xtitle=f'{config.variable:title}',
    )
    _plot_yield_comparison(**yield_args, filename=filename)
    _plot_yield_comparison(**yield_args, h_eft=h_cw_quad, eft_legend='c_{W}^{quad}=' + f'{cw:.2f}', filename=filename + '_cw')

    ### Save ###
    f_out = ROOT.TFile(f'{config.gbl.output_dir}/results/{config.base_name}.plu_yields.root', 'UPDATE')
    h_fit.Write('fit', ROOT.TObject.kOverwrite)
    h_mc.Write('SM', ROOT.TObject.kOverwrite)
    h_cw_quad.Write('cWq=0.12', ROOT.TObject.kOverwrite)
    plot.success(f'master.py::plot_plu_yields({config.base_name}) saved fit results to {f_out.GetName()}')


def plot_mc_gpr_stack(
        config : SingleChannelConfig,
        filename : str, 
        subtitle : list[str] = [], 
        mu_diboson : float = 1,
    ):
    '''
    Plots a stack plot comparing data to MC backgrounds and the GPR V+jets estimate.
    '''
    ### Adjusted bins ###
    bins = config.bins #utils.get_adjusted_bins(config.lepton_channel, config.variable)

    ### Get hists ###
    f_gpr = ROOT.TFile(f'{config.gbl.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + config.variable.name)
    h_gpr = _reassign_bins(h_gpr, bins)

    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + config.var_reader_name
    h_diboson = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.diboson, hist_name)
    h_ttbar = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, hist_name)
    h_stop = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.stop, hist_name)
    h_data = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)

    h_diboson = _reassign_bins(plot.rebin(h_diboson, config.bins), bins)
    h_ttbar = _reassign_bins(plot.rebin(h_ttbar, config.bins), bins)
    h_stop = _reassign_bins(plot.rebin(h_stop, config.bins), bins)
    h_data = _reassign_bins(plot.rebin(h_data, config.bins), bins)

    h_ttbar.Scale(config.gbl.ttbar_fitter.mu_ttbar_nom[0])
    h_stop.Scale(config.gbl.mu_stop[0])
    h_diboson.Scale(mu_diboson)

    ### Sum and ratios ###
    h_sum = h_gpr.Clone()
    h_sum.Add(h_diboson)
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)

    h_ratio = h_data.Clone()
    h_errs = h_sum.Clone()
    h_ratio.Divide(h_sum)
    h_errs.Divide(h_sum)

    stack = [
        (h_diboson.Integral(), h_diboson, 'Diboson (#mu=1)' if mu_diboson == 1 else 'Diboson', plot.colors.pastel_blue),
        (h_stop.Integral(), h_stop, 'Single top', plot.colors.pastel_yellow),
        (h_ttbar.Integral(), h_ttbar, 't#bar{t}', plot.colors.pastel_orange),
        (h_gpr.Integral(), h_gpr, 'GPR (V+jets)', plot.colors.pastel_red),
    ]
    stack.sort()

    ### Plot ###
    pads = plot.RatioPads()
    plotter1 = pads.make_plotter1(
        ytitle='Events',
        logy=True,
        subtitle=['#sqrt{s}=13 TeV, 140 fb^{-1}'] + subtitle,
        y_min=0.2,
    )
    plotter1.add(
        objs=[x[1] for x in stack],
        legend=[x[2] for x in stack],
        stack=True,
        opts='HIST',
        fillcolor=[x[3] for x in stack],
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
        objs=[h_data],
        legend=['Data'],
        opts='PE',
        legend_opts='PEL',
    )
    plotter1.draw()

    ### Subplot ###
    plotter2 = pads.make_plotter2(
        ytitle='Data / Bkgs',
        xtitle=f'{config.variable:title}',
        ignore_outliers_y=False,
        y_range=(0.5, 1.5),
    )
    plotter2.add(
        objs=[h_errs],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend=None,
    )
    plotter2.add(
        objs=[h_ratio],
        opts='PE0',
        legend=None,
    )
    plotter2.draw()
    plotter2.draw_hline(1, ROOT.kDashed)

    plot.save_canvas(pads.c, filename)


def plot_plu_fit(config : SingleChannelConfig, fit_results : dict[str, tuple[float, float]]):
    '''
    Plots a stack plot comparing data to backgrounds after the PLU fit.
    '''
    ######################################################################################
    ### HISTS
    ######################################################################################
    
    ### Adjusted bins ###
    bins = config.bins #utils.get_adjusted_bins(config.lepton_channel, config.variable)

    ### GPR ###
    f_gpr = ROOT.TFile(f'{config.gbl.output_dir}/gpr/gpr_{config.lepton_channel}lep_vjets_yield.root')
    h_gpr = f_gpr.Get('Vjets_SR_' + config.variable.name)
    h_gpr = _reassign_bins(h_gpr, bins)

    ### Response matrix ###
    f_response_mtx = ROOT.TFile(config.gbl.response_matrix_filepath.format(lep=config.lepton_channel))
    h_signals = []
    for i in range(1, len(config.bins)):
        h = f_response_mtx.Get(f'ResponseMatrix_{config.variable}_fid{i:02}')
        h.Scale(fit_results[f'mu_{i:02}'][0])
        h_signals.append(_reassign_bins(h, bins))

    ### MC + data ###
    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + config.var_reader_name
    h_ttbar = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.ttbar, hist_name)
    h_stop = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.stop, hist_name)
    h_data = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)

    h_ttbar = _reassign_bins(plot.rebin(h_ttbar, config.bins), bins)
    h_stop = _reassign_bins(plot.rebin(h_stop, config.bins), bins)
    h_data = _reassign_bins(plot.rebin(h_data, config.bins), bins)

    h_ttbar.Scale(fit_results['mu-ttbar'][0])
    h_stop.Scale(fit_results['mu-stop'][0])

    ### Sum and ratio ###
    h_sum = h_gpr.Clone()
    h_sum.Add(h_ttbar)
    h_sum.Add(h_stop)
    for h in h_signals:
        h_sum.Add(h)

    h_ratio = h_data.Clone()
    h_errs = h_sum.Clone()
    h_ratio.Divide(h_sum)
    h_errs.Divide(h_sum)

    stack = [
        (h_stop.Integral(), h_stop, 'top', plot.colors.pastel_yellow),
        (h_ttbar.Integral(), h_ttbar, 't#bar{t}', plot.colors.pastel_orange),
        (h_gpr.Integral(), h_gpr, 'V+jets', plot.colors.pastel_red),
    ]
    stack.sort()

    ######################################################################################
    ### PLOT
    ######################################################################################

    pads = plot.RatioPads()

    ### Main pad options ###
    plotter1 = pads.make_plotter1(
        text_pos='top',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel post-PLU',
        ],
        legend_columns=2,
        legend_vertical_order=True,
        ytitle='Events',
        logy=True,
        y_min=0.2,
    )

    ### Colors ###
    palette_colors = plot.colors_from_palette(ROOT.kViridis, len(h_signals))
    palette_colors = [plot.whiteblend(ROOT.gROOT.GetColor(x), 0.4) for x in palette_colors]
    bkg_colors = [x[3] for x in stack]
    def stack_color(i):
        if i < len(h_signals):
            return palette_colors[i].GetNumber()
        else:
            return bkg_colors[i - len(h_signals)]

    ### Legend ###
    legend = [f'Bin {i}' for i in range(1, len(config.bins))][::-1]
    legend += [x[2] for x in stack]

    ### Plot stack ###
    plotter1.add(
        objs=h_signals[::-1] + [x[1] for x in stack],
        legend=legend,
        stack=True,
        opts='HIST',
        fillcolor=stack_color,
        linewidth=1,
    )

    ### Plot error hash ###
    plotter1.add(
        objs=[h_sum],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend_opts='F',
        legend=['#splitline{GPR +}{MC stat}'],
    )

    ### Plot data ###
    plotter1.add(
        objs=[h_data],
        legend=['Data'],
        opts='PE',
        legend_opts='PEL',
    )
    plotter1.draw()

    ### Subplot ###
    plotter2 = pads.make_plotter2(
        ytitle='Data / Fit',
        xtitle=f'{config.variable:title}',
        ignore_outliers_y=False,
        y_range=[0.5, 1.5],
    )
    plotter2.add(
        objs=[h_errs],
        fillcolor=plot.colors.gray,
        fillstyle=3145,
        linewidth=0,
        markerstyle=0,
        opts='E2',
        legend=None,
    )
    plotter2.add(
        objs=[h_ratio],
        opts='PE0',
        legend=None,
    )
    plotter2.draw()
    plotter2.draw_hline(1, ROOT.kDashed)

    plot.save_canvas(pads.c, f'{config.gbl.output_dir}/plots/{config.base_name}.plu_postfit')


def plot_pulls(fit_results : dict[str, tuple[float, float]], filename : str, subtitle : list[str] = []):
    all_vars = [utils.variation_lumi] + utils.variations_custom + utils.variations_hist
    nvars = len(all_vars)

    ### Syst grouping ###
    do_grouping = nvars > 15
    if do_grouping:
        group_fields = [ # Prefix, name, [variations] 
            ('SysEL', 'Electron', []),
            ('SysMUON', 'Muon', []),
            ('SysTAUS', 'Tau', []),
            ('SysJET', 'Jet', []),
            ('SysFT', 'Flavor', []),
            ('', 'Other', []),
        ]
        for var in all_vars:
            for prefix,group_name,var_list in group_fields:
                if var.startswith(prefix):
                    var_list.append(var)
                    break
        all_vars = []
        for prefix,group_name,var_list in group_fields:
            all_vars.extend(var_list)

    ### Create hist ###
    h = ROOT.TH1F('h_pulls', '', nvars, 0, nvars)
    for i,var in enumerate(all_vars):
        alpha = fit_results[f'alpha_{var}']
        h.SetBinContent(i + 1, alpha[0])
        h.SetBinError(i + 1, alpha[1])
        if do_grouping:
            h.GetXaxis().SetBinLabel(i + 1, '')
        else:
            h.GetXaxis().SetBinLabel(i + 1, var)
    h.GetXaxis().LabelsOption('v')

    ### Plotter ###
    ROOT.gStyle.SetErrorX(0)
    ROOT.gStyle.SetEndErrorSize(5)
    c = ROOT.TCanvas('c1', 'c1', 1000, 800)

    plotter = plot.Plotter(
        pad=c,
        _frame=h,
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            *subtitle,
        ],
        ytitle='(#theta_{fit} - #theta_{0}) / #sigma_{#theta}',
        y_range=(-5, 5),
        bottom_margin=0.2 if do_grouping else 0.5,
        tick_length_x=0 if do_grouping else 0.015,
    )

    ### Plot objects ###
    b1 = ROOT.TBox(0, -1, nvars, 1)
    b2 = ROOT.TBox(0, -2, nvars, 2)
    b1.SetFillColor(plot.colors.pastel_green)
    b2.SetFillColor(plot.colors.pastel_yellow)
    plotter.add_primitives([b2, b1])
    plotter.add([h], opts='E1', markersize=2, markerstyle=ROOT.kFullSquare)
    plotter.draw()

    ### Group labels ###
    if do_grouping:
        def draw_line(x):
            l1 = ROOT.TLine(x, -5, x, -4.7)
            l1.Draw()
            l2 = ROOT.TLine(x, 5, x, 4.7)
            l2.Draw()
            plotter.cache.append(l1)
            plotter.cache.append(l2)
        def draw_label(start, end, name):
            t = ROOT.TLatex((start + end) / 2, -5.2, name)
            t.SetTextAngle(90)
            t.SetTextAlign(ROOT.kVAlignCenter + ROOT.kHAlignRight)
            t.Draw()
            plotter.cache.append(t)

        n_running = 0
        for i,(prefix,group_name,var_list) in enumerate(group_fields):
            if i != 0:
                draw_line(n_running)
            draw_label(n_running, n_running + len(var_list), group_name)
            n_running += len(var_list)
            
    ### Save ###
    plot.save_canvas(c, filename)
    ROOT.gStyle.SetErrorX()
    ROOT.gStyle.SetEndErrorSize(0)


def plot_correlations(roofit_results, filename : str, subtitle : list[str] = []):
    ### Get parameter lists ###
    # bins = utils.get_bins(config.lepton_channel, variable)
    # alphas = [utils.variation_lumi] + utils.variations_custom + utils.variations_hist
    # mus = [f'mu_{x:02}' for x in range(1, len(bins))]
    # pars = [f'alpha_{x}' for x in alphas] + mus

    ### All pars (custom sort) ###
    def sort_name(name):
        sname = name.replace('alpha_', '').replace('gamma_', '')
        if 'mu' in sname:
            sname = '0' + sname
        elif 'stat' in sname:
            sname = '1' + sname
        elif 'lumiNP' == sname:
            sname = '2' + sname
        return (sname, name)
    pars_all = [ sort_name(v.GetName()) for v in roofit_results.floatParsFinal() ]
    pars_all.sort()
    pars_all = [ x[1] for x in pars_all ]

    ### Pruned pars ###
    prune_threshold = 0.2
    pars_prune = []
    for i,par in enumerate(pars_all):
        if 'mu' in par:
            pars_prune.append(par)
        else:
            for j in range(len(pars_all)):
                if j == i: continue
                if abs(roofit_results.correlation(pars_all[i], pars_all[j])) > prune_threshold:
                    pars_prune.append(par)
                    break

    ### Create matrices ###
    def create_hist(pars):
        n = len(pars)
        h = ROOT.TH2F('h_cov', '', n, 0, n, n, 0, n)
        for i in range(n):
            par_label = pars[i].replace('alpha_', '').replace('gamma_', '')
            h.GetXaxis().SetBinLabel(i + 1, par_label)
            h.GetYaxis().SetBinLabel(n - i, par_label)
            for j in range(n):
                h.SetBinContent(i + 1, n - j, roofit_results.correlation(pars[i], pars[j]))
        h.GetXaxis().LabelsOption('v')
        return h
    h_all = create_hist(pars_all)
    h_prune = create_hist(pars_prune)

    ### Plot ###
    plot.create_gradient(np.array([
        [0.0,  0.0, 0.2, 1.0],
        [0.1,  0.0, 0.5, 1.0],
        [0.45, 1.0, 1.0, 1.0],
        [0.55, 1.0, 1.0, 1.0],
        [0.9,  1.0, 0.5, 0.0],
        [1.0,  1.0, 0.2, 0.0],
    ]))
    ROOT.gStyle.SetNumberContours(101)
    common_args = dict(
        text_pos='topright',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            *subtitle,
        ],
        opts='COLZ',
        z_range=[-1, 1],
        left_margin=0.3,
        bottom_margin=0.3,
    )
    plot.plot(
        objs=[h_all],
        filename=filename,
        **common_args,
    )
    plot.plot(
        objs=[h_prune],
        filename=filename + f'_prune{prune_threshold}',
        **common_args,
    )


def plot_nll(nll_file_path, signal_name, file_name : str, xtitle : str, subtitle : list[str] = []):
    ### Get values ###
    nll_file = ROOT.TFile(nll_file_path)
    nll_tree = nll_file.Get('nll')
    xs = []
    ys = []
    for entry in nll_tree:
        xs.append(getattr(entry, signal_name))
        ys.append(entry.NLL)

    ### Make graph ###
    ys = np.array(ys, dtype=float)
    ys -= np.min(ys)
    g = ROOT.TGraph(len(xs), np.array(xs, dtype=float), ys)

    ### Find 1/2 sigma ranges ###
    x_min = np.min(xs)
    x_max = np.max(xs)
    interval_1s = [x_max, x_min]
    interval_2s = [x_max, x_min]
    for i in range(1000):
        x = x_min + (x_max - x_min) / 1000 * i
        if g.Eval(x, 0, 'S') < 0.5:
            interval_1s = [min(interval_1s[0], x), max(interval_1s[1], x)]
        if g.Eval(x, 0, 'S') < 2:
            interval_2s = [min(interval_2s[0], x), max(interval_2s[1], x)]

    ### Plot ###
    mu = (interval_1s[1] + interval_1s[0]) / 2
    err = (interval_1s[1] - interval_1s[0]) / 2
    plot.notice(f'master.py::plot_nll({nll_file_path}) mu={mu:.3f} +- {err:.3f}')

    c = ROOT.TCanvas('c1', 'c1', 1000, 800)
    plotter = plot._plot(
        c=c,
        objs=[g],
        opts='CP',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            *subtitle,
            f'#mu = {mu:.3f} #pm {err:.3f}',
        ],
        ytitle='Negative Log Likelihood',
        xtitle=xtitle,
    )

    ### Sigma lines ###
    plotter.draw_hline(0.5, style=ROOT.kDashed, width=1)
    plotter.draw_hline(2, style=ROOT.kDashed, width=1)
    for x in interval_1s:
        l = ROOT.TLine(x, 0, x, 0.5)
        l.SetLineStyle(ROOT.kDashed)
        l.Draw()
        plotter.cache.append(l)
    for x in interval_2s:
        l = ROOT.TLine(x, 0, x, 2)
        l.SetLineStyle(ROOT.kDashed)
        l.Draw()
        plotter.cache.append(l)

    plot.save_canvas(c, file_name)


def plot_ranking(config: MultiChannelConfig, ranking_outputs, file_name: str):
    res = subprocess.run([
            'root', '-b', '-l', '-q',
            f'drawRankingPlot.C("../../{ranking_outputs}", 0, 0, 20, "../../{file_name}")', # doPrefit, doRelative, npMax
        ],
        cwd=config.gbl.npcheck_dir,
        capture_output=True, 
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        res.check_returncode()


##########################################################################################
###                                       CONFIG                                       ###
##########################################################################################

class GlobalConfig:
    '''
    Configuration parameters common to all channels
    '''
    npcheck_dir = 'ResonanceFinder/NPCheck'
    response_matrix_format = '/response_matrix/{sample}_{lep}lep_rf_histograms.root'
    rebinned_hists_format = '/rebin/{lep}lep_{sample}_rebin.root'
    
    def __init__(
            self,
            ### Basic config ###
            file_manager : utils.FileManager,
            ttbar_fitter: ttbar_fit.TtbarSysFitter,
            mu_stop : tuple[float, float],
            output_dir : str,
            ### Run management ###
            nominal_only : bool,
            skip_hist_gen : bool,
            skip_fits : bool,
            skip_direct_fit : bool,
            skip_gpr : bool,
            skip_gpr_if_present : bool,
            skip_plu : bool,
            skip_diboson : bool,
            skip_eft : bool,
            gpr_condor : bool,
            is_asimov : bool,
            run_plu_val : bool,
        ):
        ### Basic config ###
        self.file_manager = file_manager
        self.ttbar_fitter = ttbar_fitter
        self.mu_stop = mu_stop
        self.output_dir = output_dir

        ### Output paths ###
        self.response_matrix_filepath = output_dir + GlobalConfig.response_matrix_format.format(sample='diboson', lep='{lep}')
        self.rebinned_hists_filepath = output_dir + GlobalConfig.rebinned_hists_format
        self.resonance_finder_outdir = f'{output_dir}/rf'
        os.makedirs(f'{self.output_dir}/rebin', exist_ok=True)
        os.makedirs(f'{self.output_dir}/response_matrix', exist_ok=True)
        os.makedirs(f'{self.output_dir}/results', exist_ok=True)
        os.makedirs(f'{self.output_dir}/rf', exist_ok=True)

        ### Run management ###
        self.nominal_only = nominal_only
        self.skip_hist_gen = skip_hist_gen
        self.skip_fits = skip_fits
        self.skip_direct_fit = skip_direct_fit or skip_fits
        self.skip_gpr = skip_gpr or skip_fits
        self.skip_gpr_if_present = skip_gpr_if_present
        self.skip_plu = skip_plu or skip_fits
        self.skip_diboson = skip_diboson or skip_fits
        self.skip_eft = skip_eft or skip_fits
        self.gpr_condor = gpr_condor
        self.is_asimov = is_asimov
        self.plu_validation_iters = 100 if run_plu_val else 0


class MultiChannelConfig():
    '''
    Contains configuration parameters for an aggregate of run channels, or potentially
    just a single channel.

    @property gbl
        Reference to a [GlobalConfig].
    @property lepton_tag/variable_tag
        Naming tags for the lepton/variable configuration.
    @property base_name
        Base name containing the lepton/variable configuration.
    @property variables
        Dictionary of variables per lepton channel.
    '''
    def __init__(
            self,
            global_config : GlobalConfig,
            lepton_channels : list[int],
            variable_tag : str = 'vv_m-mT',
            variables : dict[int, list[utils.Variable]] = None,
        ):
        '''
        @param variables/variable_tag
            Specify a dict of variables for each lepton channel, or a tag for a predefined
            setup (see [_parse_variable_tag]). The dictionary always takes precedence if 
            set.
        '''
        self.gbl = global_config
        self.lepton_channels = lepton_channels
        self.lepton_tag = '-'.join(str(x) for x in lepton_channels)
        if variables is not None:
            self.variables = variables
            self.variable_tag = '-'.join(str(x) for k,v in self.variables.items() for x in v)
        else:
            self.variables = MultiChannelConfig._parse_variable_tag(variable_tag, lepton_channels)
            self.variable_tag = variable_tag

        self.base_name = f'{self.lepton_tag}lep_{self.variable_tag}'
        if len(self.lepton_channels) > 1:
            self.lep_title = f'{self.lepton_tag} lepton channels'
        else: 
            self.lep_title = f'{self.lepton_tag}-lepton channel'


    def _parse_variable_tag(tag, lepton_channels) -> dict[int, list[utils.Variable]]:
        if tag == 'vv_m-mT':
            out = {
                0: [utils.Variable.vv_mt],
                1: [utils.Variable.vv_m],
                2: [utils.Variable.vv_m],
            }
        elif tag == 'vv_mT':
            out = {
                0: [utils.Variable.vv_mt],
                1: [utils.Variable.vv_mt],
            }
        elif tag == 'ptJ':
            out = {
                0: [utils.Variable.fatjet_pt],
                1: [utils.Variable.fatjet_pt],
                2: [utils.Variable.fatjet_pt],
            }
        else:
            raise NotImplementedError(f'ChannelConfig unknown tag {tag}')
        return { k:v for k,v in out.items() if k in lepton_channels }
    

    def get_variables(self, lepton_channel : int) -> list[utils.Variable]:
        return self.variables[lepton_channel]


    def split_config(self) -> list[SingleChannelConfig]:
        '''
        Splits out configs for the individual channels.
        '''
        out = []
        for lep,variables in self.variables.items():
            for var in variables:
                out.append(SingleChannelConfig(
                    global_config=self.gbl,
                    lepton_channel=lep,
                    variable=var,
                ))
        return out

    def __format__(self, format_spec: str) -> str:
        if format_spec == 'lep_title':
            return self.lep_title
        return self.base_name


class SingleChannelConfig(MultiChannelConfig):
    '''
    Contains configuration parameters for a single run channel, i.e. a lepton channel and
    variable combination. Used primarily for GPR and PLU fits. Exposes simpler properties
    than [MultiChannelConfig].
    '''
    def __init__(
            self, 
            global_config : GlobalConfig, 
            lepton_channel : int,
            variable : utils.Variable,
        ):
        super().__init__(global_config, [lepton_channel], variables={lepton_channel: [variable]})
        self.lepton_channel = lepton_channel
        self.variable = variable
        self.var_reader_name = utils.generic_var_to_lep(variable, lepton_channel).name
        self.bins = np.array(utils.get_bins(lepton_channel, variable), dtype=float)

        ### Set by run_gpr ###
        self.gpr_nominal_config : gpr.FitConfig = None
        self.gpr_results : gpr.FitResults = None
        self.gpr_sigcontam_corrs : list[float] = None


##########################################################################################
###                                  RESONANCE FINDER                                  ###
##########################################################################################


def run_npcheck_drawfit(config : SingleChannelConfig, ws_path : str):
    '''
    DEPRECATED.

    Runs drawPostFit.C from NPCheck. This is supplanted by [plot_plu_fit].

    Uses the default fccs file in the NPCheck dir, so must call in sync with the fit!
    '''
    plot.notice(f'Drawing PLU fits using NPCheck')
    with open(f'{config.gbl.output_dir}/rf/log.{config.base_name}.draw_fit.txt', 'w') as f:
        res = subprocess.run(
            ['./runDrawFit.py', ws_path,
                '--mu', '1',
                '--fccs', 'fccs/FitCrossChecks.root'
            ],
            cwd=config.gbl.npcheck_dir,
            stdout=f,
            stderr=f,
            # capture_output=True,
            # text=True,
        )
    res.check_returncode()
    npcheck_output_path = f'{config.gbl.npcheck_dir}/Plots/PostFit/summary_postfit_doAsimov0_doCondtional0_mu1.pdf'
    target_path = f'{config.gbl.output_dir}/rf/{config.base_name}.plu_postfit.pdf'
    shutil.copyfile(npcheck_output_path, target_path)


def run_rf(config : MultiChannelConfig, mode : str, skip_fits : bool, stat_validation_index : int = None):
    '''
    Runs the resonance finder script. Common to all likelihood fits.

    @param mode
        Same options as in [rf.run].
    @param skip_fits
        Set to true to skip the fits but still fetch the results from the saved files.

    @returns (ws_path, roofit_results, dict_results)
        - ws_path: path to the generated ResonanceFinder workspace.
        - roofit_results: fit results from RooFit.
        - dict_results: the RooFit results parsed into a dictionary of name: (val, err).
    '''
    # Really important this import is here otherwise segfaults occur, due to the
    # `from ROOT import RF` line I think. But somehow hiding it here is fine.
    import rf_plu

    ### Create RF workspace ###
    if not skip_fits:
        plot.notice(f'master.py::run_rf({config.base_name}) creating ResonanceFinder workspace for {mode} fit')
        ws_path = rf_plu.run(
            mode=mode,
            variables=config.variables,
            response_matrix_path=config.gbl.response_matrix_filepath,
            output_dir=config.gbl.output_dir,
            base_name=config.base_name,
            hist_file_format=config.gbl.rebinned_hists_filepath,
            mu_stop=config.gbl.mu_stop,
            mu_ttbar=config.gbl.ttbar_fitter.mu_ttbar_nom,
            gpr_mu_corrs=not config.gbl.nominal_only,
            stat_validation_index=stat_validation_index,
        )
    else:
        ws_path = rf_plu.ws_path(
            mode=mode,
            output_dir=config.gbl.output_dir, 
            base_name=config.base_name,
            stat_validation_index=stat_validation_index,
        )
    ws_path = os.path.abspath(ws_path)

    ### Run fits ###
    if stat_validation_index is None:
        fcc_path = f'{config.gbl.output_dir}/rf/{config.base_name}.{mode}_fcc.root'
    else:
        fcc_path = f'{config.gbl.output_dir}/rf/{config.base_name}.{mode}_fcc_var{stat_validation_index:03}.root'
    if not skip_fits:
        plot.notice(f'master.py::run_rf({config.base_name}) running {mode} fits')
        with open(f'{config.gbl.output_dir}/rf/log.{config.base_name}.{mode}_fcc.txt', 'w') as f:
            res = subprocess.run(
                ['./runFitCrossCheck.py', ws_path],  # the './' is necessary!
                cwd=config.gbl.npcheck_dir,
                stdout=f,
                stderr=f
            )
        res.check_returncode()
        shutil.copyfile(f'{config.gbl.npcheck_dir}/fccs/FitCrossChecks.root', fcc_path)

    ### Get fit result ###
    fcc_file = ROOT.TFile(fcc_path)
    results_name = 'PlotsAfterGlobalFit/unconditionnal/fitResult'
    roofit_results = fcc_file.Get(results_name)
    if not skip_fits:
        roofit_results.Print()

    ### Parse results ###
    dict_results = { v.GetName() : (v.getValV(), v.getError()) for v in roofit_results.floatParsFinal() }
    dict_results['mu-stop'] = convert_alpha(dict_results['alpha_mu-stop'], config.gbl.mu_stop)
    dict_results['mu-ttbar'] = convert_alpha(dict_results['alpha_mu-ttbar'], config.gbl.ttbar_fitter.mu_ttbar_nom)

    return ws_path, roofit_results, dict_results


def run_plu(config : SingleChannelConfig, stat_validation_index : int = None):
    '''
    Runs the profile likelihood unfolding fit using ResonanceFinder. Assumes the
    GPR/response matrices have been created already.
    '''
    try:
        ws_path, roofit_results, plu_fit_results = run_rf(config, 'PLU', config.gbl.skip_plu, stat_validation_index)

        ### Plots ###
        if stat_validation_index is None:
            ### Draw fit ###
            plot_plu_fit(config, plu_fit_results)

            ### Draw pulls ###
            plot_pulls(
                fit_results=plu_fit_results, 
                filename=f'{config.gbl.output_dir}/plots/{config.base_name}.plu_pulls',
                subtitle=[f'{config.lepton_channel}-lepton channel {config.variable.title} pulls'],
            )

            ### Draw correlation matrix ###
            plot_correlations(
                roofit_results=roofit_results, 
                filename=f'{config.gbl.output_dir}/plots/{config.base_name}.plu_corr',
                subtitle=[f'{config.lepton_channel}-lepton channel {config.variable.title} fit'],
                )

            ### Draw yield vs MC ###
            plot_plu_yields(config, plu_fit_results, f'{config.gbl.output_dir}/plots/{config.base_name}.plu_yields')

        return plu_fit_results
    except Exception as e:
        if config.gbl.skip_plu:
            plot.warning(f'master.py::run_plu({config}) skipping, caught error:\n\t{e}')
        else:
            raise e


def run_plu_val(config : SingleChannelConfig, results_nom : dict[str, tuple[float, float]]):
    '''
    Runs a validation test for the PLU fit by creating statistical variations of the data
    histograms (see [save_data_variation_histograms]) and running the PLU on each of them,
    generating a distribution of the output results.
    '''
    nbins = len(config.bins) - 1

    ### Create histograms ###
    hists = []
    for i in range(nbins):
        hists.append(ROOT.TH1F(f'h_plu_validation_{i}', '', 20, -5, 5))
    sums = np.zeros(nbins)
    sum_squares = np.zeros(nbins)

    ### Run PLU multiple times ###
    for val_index in range(config.gbl.plu_validation_iters):
        results = run_plu(config, val_index)
        for i in range(nbins):
            name = f'mu_{i + 1:02}'
            val_nom = results_nom[name]
            val = (results[name][0] - val_nom[0]) / val_nom[1]

            hists[i].Fill(val)
            sums[i] += val
            sum_squares[i] += val**2

    ### Mean and std dev callback ###
    means = sums / config.gbl.plu_validation_iters
    std_devs = np.sqrt(sum_squares / config.gbl.plu_validation_iters - means**2)
    def callback(plotter):
        for i in range(nbins):
            line = ROOT.TLine(means[i], i, means[i], i + 0.5)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line.Draw()

            line2 = ROOT.TLine(means[i] - std_devs[i], i, means[i] - std_devs[i], i + 0.5)
            line2.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line2.SetLineStyle(ROOT.kDashed)
            line2.Draw()

            line3 = ROOT.TLine(means[i] + std_devs[i], i, means[i] + std_devs[i], i + 0.5)
            line3.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(2)
            line3.SetLineStyle(ROOT.kDashed)
            line3.Draw()

            plotter.cache.extend([line, line2, line3])

    ### Legend for mean lines ###
    line_mean = ROOT.TLine(0, 0, 1, 1)
    line_mean.SetLineColor(ROOT.kBlack)
    line_mean.SetLineWidth(2)

    line_var = ROOT.TLine(0, 0, 1, 1)
    line_var.SetLineColor(ROOT.kBlack)
    line_var.SetLineWidth(2)
    line_var.SetLineStyle(ROOT.kDashed)
    legend = [
        (line_mean, 'Mean', 'L'),
        (line_var, '#pm1 std. dev.', 'L'),
    ]

    ### Plot ###
    plot.plot_tiered(
        hists=[[h] for h in hists],
        text_pos='top',
        tier_labels=[f'Bin {i:02}' for i in range(1, len(config.bins))],
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'PLU fit results from Poisson varied data',
        ],
        legend=legend,
        xtitle='Fitted Fiducial Events #frac{x - x_{nom}}{#sigma_{nom}}',
        text_offset_top = 0.02,
        y_pad_top=0.2,
        title_offset_x=1.5,
        bottom_margin=0.2,
        fillcolor=plot.colors.pastel,
        callback=callback,
        filename=f'{config.gbl.output_dir}/plots/{config.base_name}.plu_validation',
    )


def run_diboson_fit(config : MultiChannelConfig, skip_fits : bool = False):
    '''
    Runs the resonance finder script to fit the diboson cross section in all channels. 
    '''
    try:
        ws_path, roofit_results, dict_results = run_rf(config, 'diboson', skip_fits)
        mu_diboson = dict_results['mu-diboson']
        
        ### Plot fit ###
        # TODO this doesn't adjust the GPR distribution for the correlation correction
        sc_configs = config.split_config()
        for sc_config in sc_configs:
            filename = f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_postfit'
            if len(sc_configs) > 1:
                filename += f'_{sc_config.base_name}'
            plot_mc_gpr_stack(
                config=sc_config, 
                subtitle=[
                    f'{sc_config.lepton_channel}-lepton channel postfit',
                    f'Diboson #mu = {mu_diboson[0]:.2f} #pm {mu_diboson[1]:.2f}',
                ],
                mu_diboson=mu_diboson[0],
                filename=filename,
            )

        ### Draw pulls ###
        plot_pulls(
            fit_results=dict_results, 
            filename=f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_pulls',
        )

        ### Draw correlation matrix ###
        plot_correlations(
            roofit_results=roofit_results, 
            filename=f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_corr',
        )

        ### NLL ###
        if not skip_fits:
            plot.notice(f'master.py::run_diboson_fit() Running {config.base_name} diboson-xsec NLL')
            run_nll(config.gbl.output_dir, f'{config.base_name}.diboson', ws_path, asimov=False)
            run_nll(config.gbl.output_dir, f'{config.base_name}.diboson', ws_path, asimov=True)
        plot_nll(
            f'{config.gbl.output_dir}/rf/{config.base_name}.diboson_nll.root',
            file_name=f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_nll',
            signal_name='mu-diboson',
            xtitle='#mu(diboson)',
        )
        plot_nll(
            f'{config.gbl.output_dir}/rf/{config.base_name}.diboson_nll-asimov.root',
            file_name=f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_nll-asimov',
            signal_name='mu-diboson',
            xtitle='#mu(diboson)',
        )

        ### Ranking ###
        if not skip_fits:
            plot.notice(f'master.py::run_diboson_fit() Running {config.base_name} diboson-xsec ranking')
            run_ranking(config, f'{config.base_name}.diboson', ws_path)
        plot_ranking(
            config=config,
            ranking_outputs=f'{config.gbl.output_dir}/rf/ranking/{config}.diboson',
            file_name=f'{config.gbl.output_dir}/plots/{config}.diboson_ranking',
        )

    except Exception as e:
        if skip_fits:
            plot.warning(f'master.py::run_diboson_fit({config.base_name}) skipping, caught error {e}')
        else:
            raise e


def run_nll(output_dir : str, base_name : str, ws_path : str, granularity=5, asimov=False, mu=1, range_sigmas=3):
    '''
    Calls the NLL macro in NPCheck. This calls the python runner script as a subprocess
    for simplicity, but also note the macro is side-effectful so probably can't use
    directly anyways.

    @param range_sigmas
        The number of standard deviations (as determined from the minimization) of the POI
        up and down to scan over. [granularity] evenly spaced points will be tested from
        this range.
    '''
    log_name = f'{output_dir}/rf/log.{base_name}_nll' + ('-asimov' if asimov else '') + '.txt'
    with open(log_name, 'w') as f:
        # For some reason passing the fccs file breaks this
        res = subprocess.run(
            [
                './runNLL.py', ws_path, # the './' is necessary!
                '--granularity', str(granularity), 
                '--doAsimov' if asimov else '--no-doAsimov', 
                '--mu', str(mu), 
                '--range', str(range_sigmas),
                '--no-doPlot',
            ],  
            cwd='ResonanceFinder/NPCheck',
            stdout=f,
            stderr=f,
        )
    res.check_returncode()
    shutil.copyfile(
        'ResonanceFinder/NPCheck/nllscan/nll.root', 
        f'{output_dir}/rf/{base_name}_nll' + ('-asimov' if asimov else '') + '.root',
    )
    # shutil.copyfile(
    #     'ResonanceFinder/NPCheck/Plots/NLL/summary.pdf',
    #     f'{output_dir}/plots/{base_name}_nll' + ('-asimov' if asimov else '') + '.pdf',
    # )


def run_ranking(config : MultiChannelConfig, base_name: str, ws_path: str, asimov: bool = False):
    '''
    Calls the ranking macro in NPCheck. This calls the python runner script as a subprocess
    for simplicity, but also note the macro is side-effectful so probably can't use
    directly anyways.
    '''
    # outdir = f'../../{output_dir}/rf/ranking/{base_name}'
    # res = subprocess.run(
    #     [
    #         './runRanking.py', ws_path, # the './' is necessary!
    #         '--outname', outdir,
    #         '--fccs', ws_path.replace('_ws', '_fcc'),
    #         '--doAsimov' if asimov else '--no-doAsimov', 
    #         '--condor',
    #     ],  
    #     cwd='ResonanceFinder/NPCheck',
    #     capture_output=True,
    #     text=True,
    # )
    # res.check_returncode()
    # print(res.stdout)

    condor = RankingSubmitMaker(
        config=config.gbl,
        filepath=f'{config.gbl.output_dir}/rf/{base_name}_ranking.submit',
    )
    condor.add(
        systs=[f'alpha_{x}' for x in utils.variations_custom + utils.variations_hist + [utils.variation_lumi]], 
        ws_path=ws_path,
        fcc_path=ws_path.replace('_ws', '_fcc'),
        output_path=f'{config.gbl.output_dir}/rf/ranking/{base_name}',
        asimov=asimov,
    )
    condor.close()
    condor.run()
    condor.wait()


def run_eft_fit(config : MultiChannelConfig, mode : str, skip_fits : bool = False):
    '''
    Runs a ResonanceFinder fit to the EFT Wilson coefficients. 
    '''
    ### Parsing ###
    operator, order = mode.split('_')
    if operator == 'cw':
        operator_title = 'c_{W}'
    else:
        raise NotImplementedError(f'master.py::run_eft_fit() Unknown operator {operator}')

    ### Fit ###
    try:
        ws_path, roofit_results, dict_results = run_rf(config, mode, skip_fits)

        ### Plot fit ###
        # sc_configs = config.split_config()
        # filename = f'{config.gbl.output_dir}/plots/{config.base_name}.diboson_postfit'
        # for sc_config in sc_configs:
        #     if len(sc_configs) > 1:
        #         filename += f'_{sc_config.base_name}'
        #     plot_mc_gpr_stack(
        #         config=config, 
        #         subtitle=[
        #             f'{sc_config.lepton_channel}-lepton channel postfit',
        #             f'Diboson #mu = {mu_diboson[0]:.2f} #pm {mu_diboson[1]:.2f}',
        #         ],
        #         mu_diboson=mu_diboson[0],
        #         filename=filename,
        #     )

        ### Draw pulls ###
        plot_pulls(
            fit_results=dict_results, 
            filename=f'{config.gbl.output_dir}/plots/{config.base_name}.{mode}_pulls',
            subtitle=[f'{config.lep_title} {mode} pulls'],
        )

        ### Draw correlation matrix ###
        plot_correlations(
            roofit_results=roofit_results, 
            filename=f'{config.gbl.output_dir}/plots/{config.base_name}.{mode}_corr',
            subtitle=[f'{config.lep_title} {mode} fit'],
        )

        ### Run NLL ###
        # For some reason the quad asimov gets +- 0.000 and the script breaks
        # These also take really long so just skip the asimov nll.
        if not skip_fits:
            plot.notice(f'master.py::run_eft_fit() Running {config.base_name} {mode} NLL')
            run_nll(config.gbl.output_dir, f'{config.base_name}.{mode}', ws_path, asimov=False, mu=0, range_sigmas=1 if order == 'quad' else 3, granularity=11)
            # run_nll(config.gbl.output_dir, f'{config.base_name}.{mode}', ws_path, asimov=True, mu=0, range_sigmas=1.5 if order == 'quad' else 3, granularity=11)
        plot_nll(
            f'{config.gbl.output_dir}/rf/{config.base_name}.{mode}_nll.root',
            file_name=f'{config.gbl.output_dir}/plots/{config.base_name}.{mode}_nll',
            subtitle=[f'{config:lep_title} {mode} fit'],
            signal_name=f'mu-{operator}',
            xtitle=operator_title,
        )
        # plot_nll(
        #     f'{config.gbl.output_dir}/rf/{config.base_name}.{mode}_nll-asimov.root',
        #     file_name=f'{config.gbl.output_dir}/plots/{config.base_name}.{mode}_nll-asimov',
        #     subtitle=[f'{config:lep_title} {mode} fit'],
        #     signal_name=f'mu-{operator}',
        #     xtitle=operator_title,
        # )
    except Exception as e:
        if skip_fits:
            plot.warning(f'master.py::run_eft_fit({mode}) skipping, caught error {e}')
        else:
            raise e


##########################################################################################
###                                         RUN                                        ###
##########################################################################################


def save_rebinned_histograms(config : SingleChannelConfig):
    '''
    Rebins the CxAODReader histograms to match the response matrix/gpr.
    '''
    ### Output ###
    output_files = {}
    def file(sample):
        if sample not in output_files:
            output_files[sample] = ROOT.TFile(config.gbl.rebinned_hists_filepath.format(lep=config.lepton_channel, sample=sample), 'UPDATE')
        return output_files[sample]

    ### All histo variations ###
    variations = [utils.variation_nom]
    for x in utils.variations_hist:
        variations.append(x + utils.variation_up_key)
        variations.append(x + utils.variation_down_key)

    ### Save ###
    for variation in variations:
        for channel in ['SR', 'TCR']:
            if channel == 'TCR' and config.lepton_channel != 1: continue
            hist_name = '{sample}_VV{lep}_MergHP_Inclusive_' + f'{channel}_{config.var_reader_name}'
            hists = config.gbl.file_manager.get_hist_all_samples(config.lepton_channel, hist_name, variation)
            for sample_name,hist in hists.items():
                if sample_name == utils.Sample.data.name: continue # Handled in [save_data_variation_histograms]

                hist = plot.rebin(hist, config.bins)
                new_name = hist_name.format(lep=f'{config.lepton_channel}Lep', sample=sample_name)
                new_name = utils.hist_name_variation(new_name, config.gbl.file_manager.samples[sample_name], variation, separator='__')
                # For some reason RF expects a double underscore...not sure how to change
                hist.SetName(new_name)

                f = file(sample_name)
                f.cd()
                hist.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)

    ### Data rebinned histograms for PLU stat test ###
    save_data_variation_histograms(config, file(utils.Sample.data.name))

    plot.success(f'Saved rebinned histograms to {config.gbl.rebinned_hists_filepath}')


def save_data_variation_histograms(config : SingleChannelConfig, f : ROOT.TFile):
    '''
    Saves several statistical variations of the data histograms
    '''
    ### Config ###
    hist_name = '{sample}_VV{lep}_MergHP_Inclusive_SR_' + config.var_reader_name

    ### Nominal hist ###
    h_nom = config.gbl.file_manager.get_hist(config.lepton_channel, utils.Sample.data, hist_name)
    h_nom = plot.rebin(h_nom, config.bins)
    h_nom.SetName(hist_name.format(lep=f'{config.lepton_channel}Lep', sample='data'))

    f.cd()
    h_nom.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)

    ### Create variations ###
    rng = np.random.default_rng()
    for i in range(config.gbl.plu_validation_iters):
        name = hist_name.format(lep=f'{config.lepton_channel}Lep', sample=f'data_var{i:03}')
        h = ROOT.TH1F(name, name, len(config.bins) - 1, config.bins)
        for x in range(1, h_nom.GetNbinsX() + 1):
            h[x] = rng.poisson(h_nom[x])
        h.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)


def make_gpr_floating_correlation_hists(config : SingleChannelConfig):
    '''
    Given a floating parameter like mu_diboson, the correlation with the GPR yield is
    implemented as an additive correction histogram (1 - mu) * diff. But ResonanceFinder
    can't scale by (1 - mu), so we have to break it apart into diff + mu * (-diff).

    @requires
        [config.gpr_sigcontam_corrs] to be set.
    '''
    bins = utils.get_bins(config.lepton_channel, config.variable)
    h_diff = ROOT.TH1F(f'gpr_mu-diboson_posdiff_{config.variable}', '', len(bins) - 1, np.array(bins, dtype=float))
    for i,v in enumerate(config.gpr_sigcontam_corrs):
        h_diff.SetBinContent(i+1, v)
        h_diff.SetBinError(i+1, 0)

    h_neg = h_diff.Clone()
    h_neg.SetName(f'gpr_mu-diboson_negdiff_{config.variable}')
    h_neg.Scale(-1)

    f_output = ROOT.TFile(config.gpr_nominal_config.output_root_file_path, 'UPDATE')
    h_diff.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)
    h_neg.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)
    f_output.Close()


def run_gpr(channel_config : SingleChannelConfig):
    '''
    Runs the GPR fit for every variation.
    '''
    plot.notice(f'master.py::run_gpr() Running GPR fits for {channel_config.base_name}')

    ### Condor ###
    if channel_config.gbl.gpr_condor:
        condor_file = GprSubmitMaker(channel_config.gbl, f'{channel_config.gbl.output_dir}/gpr/{channel_config.base_name}.submit.condor')

    ### Config ###
    def make_config(variation, mu_stop):
        return gpr.FitConfig(
            lepton_channel=channel_config.lepton_channel,
            var=channel_config.variable,
            output_hists_dir=f'{channel_config.gbl.output_dir}/gpr',
            output_plots_dir=f'{channel_config.gbl.output_dir}/gpr/{channel_config.lepton_channel}lep/{channel_config.variable}/{variation}',
            use_vjets_mc=channel_config.gbl.is_asimov,
            variation=variation,
            mu_ttbar=channel_config.gbl.ttbar_fitter.get_var(variation),
            mu_stop=mu_stop,
        )
    mu_diboson_points = [0.9, 0.95, 1.05, 1.1]

    ### Summary plots ###
    def summary_actions():
        fit_config = make_config(utils.variation_nom, channel_config.gbl.mu_stop)
        channel_config.gpr_nominal_config = fit_config
        channel_config.gpr_results = fit_config.fit_results
        if not channel_config.gbl.nominal_only:
            channel_config.gpr_sigcontam_corrs = plot_gpr_mu_diboson_correlations(
                config=fit_config,
                yields=mu_diboson_points,
                filename=f'{channel_config.gbl.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_diboson_mu_scan',
            )
            plot_gpr_ttbar_and_stop_correlations(fit_config, f'{channel_config.gbl.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_ttbar_stop_mu_scan')
            plot_gpr_mc_comparisons(channel_config, f'{channel_config.gbl.output_dir}/plots/{fit_config.lepton_channel}lep_{fit_config.var}.gpr_mc_comparison')

            ### Get diboson (1 - mu) histograms ###
            make_gpr_floating_correlation_hists(channel_config)
    

    if channel_config.gbl.skip_fits or channel_config.gbl.skip_gpr:
        summary_actions()
        return

    ### Common run function ###
    def run(variation, mu_stop=channel_config.gbl.mu_stop[0]):
        config = make_config(variation, mu_stop)
        if channel_config.gbl.gpr_condor:
            condor_file.add(config)
        else:
            gpr.run(
                file_manager=channel_config.gbl.file_manager,
                config=config,
                from_csv_only=channel_config.gbl.skip_fits or channel_config.gbl.skip_gpr,
                skip_if_in_csv=channel_config.gbl.skip_gpr_if_present,
            )

    ### Nominal ###
    run('nominal')
    if channel_config.gbl.nominal_only:
        summary_actions()
        return

    ### Diboson signal strength variations ###
    for mu_diboson in mu_diboson_points:
        run(f'mu-diboson{mu_diboson}')

    ### ttbar signal strength variations ###
    for updown in [utils.variation_up_key, utils.variation_down_key]:
        run(utils.variation_mu_ttbar + updown)

    ### Single top signal strength variations ###
    run(
        variation=utils.variation_mu_stop + utils.variation_up_key,
        mu_stop=channel_config.gbl.mu_stop[0] + channel_config.gbl.mu_stop[1],
    )
    run(
        variation=utils.variation_mu_stop + utils.variation_down_key,
        mu_stop=channel_config.gbl.mu_stop[0] - channel_config.gbl.mu_stop[1],
    )

    ### Syst variations ###
    for variation_base in utils.variations_hist:
        for updown in [utils.variation_up_key, utils.variation_down_key]:
            run(variation_base + updown)

    ### Condor ###
    if channel_config.gbl.gpr_condor:
        condor_file.close()
        condor_file.run()
        # plot.success("Launched GPR jobs on condor. Once jobs are done, merge the results using merge_gpr_condor.py, then recall master.py using --skip-gpr.")
        condor_file.wait()
        merge_gpr_condor.merge(channel_config.gbl.output_dir)
    
    ### Outputs ###
    summary_actions()


def run_direct_fit(config : SingleChannelConfig):
    '''
    Test fit where we fit the detector-level diboson signal strength to each bin directly.

    This will be superceded by the profile likelihood fit, but is a nice check.
    '''
    bins = utils.get_bins(config.lepton_channel, config.variable)
    h_diboson_fit = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
    h_diboson_mc = ROOT.TH1F('h_diboson', 'Diboson', len(bins) - 1, np.array(bins, dtype=float))
    for i in range(len(bins) - 1):
        res = diboson_fit.run_fit(
            config=config,
            bin=(bins[i], bins[i+1]),
            gpr_mu_corr=config.gpr_sigcontam_corrs[i] if config.gpr_sigcontam_corrs is not None else None,
        )
        diboson_yield = res['diboson-yield']
        h_diboson_fit.SetBinContent(i+1, diboson_yield[0])
        h_diboson_fit.SetBinError(i+1, diboson_yield[1])

        diboson_yield = res['diboson-yield-mc-statonly']
        h_diboson_mc.SetBinContent(i+1, diboson_yield[0])
        h_diboson_mc.SetBinError(i+1, diboson_yield[1])
    _plot_yield_comparison(
        h_fit=h_diboson_fit,
        h_mc=h_diboson_mc,
        filename=f'{config.output_dir}/plots/{config.base_name}.directfit_yields',
        subtitle=[
            '#sqrt{s}=13 TeV, 140 fb^{-1}',
            f'{config.lepton_channel}-lepton channel',
            'Direct bin-by-bin fit',
        ],
        xtitle=f'{config.variable:title}',
    )


def run_single_channel(config : SingleChannelConfig):
    '''
    Main run function for a single lepton/variable channel.
    '''
    if not config.gbl.skip_hist_gen and not config.gbl.skip_fits and not config.gbl.skip_gpr:
        ### Generate response matricies ###
        plot.notice(f'master.py::run_channel({config.base_name}) creating response matrix')
        unfolding.main(
            file_manager=config.gbl.file_manager,
            sample=utils.Sample.diboson,
            lepton_channel=config.lepton_channel,
            output=f'{config.gbl.output_dir}/response_matrix',
            output_plots=f'{config.gbl.output_dir}/plots',
            vars=[config.variable],
        )

        ### Rebin reco histograms ###
        save_rebinned_histograms(config)

    ### GPR fit ###
    gc.collect() # Get segfaults when generating plots sometimes
    gc.disable() # https://root-forum.cern.ch/t/segfault-on-creating-canvases-and-pads-in-a-loop-with-pyroot/44729/13
    run_gpr(config) # When skip_gpr, still generates the summary plots
    gc.enable()

    ### Prefit plot (pre-likelihood fits but using GPR) ###
    plot_mc_gpr_stack(
        config=config,
        subtitle=[f'{config.lepton_channel}-lepton channel prefit (post-GPR)'],
        filename=f'{config.gbl.output_dir}/plots/{config.base_name}.prefit',
    )

    ### Naive yields ###
    plot_naive_bin_yields(
        config=config,
        filename=f'{config.gbl.output_dir}/plots/{config.base_name}.naive_yields'
    )

    ### Diboson yield ###
    if not config.gbl.skip_direct_fit:
        run_direct_fit(config)

    ### Resonance finder fits ###
    try: # This requires ResonanceFinder!
        ### PLU fit ###
        plu_results = run_plu(config)

        ### PLU validation test ###
        if config.gbl.plu_validation_iters:
            run_plu_val(config, plu_results)
        
        ### Diboson fit ###
        run_diboson_fit(config, skip_fits=config.gbl.skip_diboson)

        ### EFT fits ###
        run_eft_fit(config, 'cw_lin', skip_fits=config.gbl.skip_eft)
        run_eft_fit(config, 'cw_quad', skip_fits=config.gbl.skip_eft)

    except Exception as e:
        plot.error(str(e))
        raise e


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="Master run script for doing the unfolded analysis for VVsemilep.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('filepaths', nargs='+')
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('--skip-hist-gen', action='store_true', help="Skip generating the response matrix and rebinned histograms.")
    parser.add_argument('--skip-fits', action='store_true', help="Don't do any fits. For GPR, uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'. For the others, will look for the ResonanceFinder fcc files at '{output}/rf/*_fcc.root'")
    parser.add_argument('--skip-gpr', action='store_true', help="Skip only the GPR fits; uses the fit results stored in the CSV. This file should be placed at '{output}/gpr/gpr_fit_results.csv'.")
    parser.add_argument('--skip-plu', action='store_true', help="Skips the PLU fit.")
    parser.add_argument('--skip-diboson', action='store_true', help="Skips the diboson cross section fit.")
    parser.add_argument('--skip-eft', action='store_true', help="Skips the EFT fits.")
    parser.add_argument('--fit-bin-yields', action='store_true', help="Also do direct bin-by-bin diboson yield fits (old, diagnostic test).")
    parser.add_argument('--rerun-gpr', action='store_true', help="By default, will skip GPR fits that are present in the CSV already. This flag will force a rerun of all bins.")
    parser.add_argument('--skip-channels', action='store_true', help="Skips all per-channel processing, and jumps to the multichannel fits.")
    parser.add_argument('--nominal', action='store_true', help="Run without any correlations to other signal strengths or systematic variations.")
    parser.add_argument('--no-systs', action='store_true', help="Run without using the systematic variations, but still does the ttbar/diboson mu adjustments.")
    parser.add_argument('--asimov', action='store_true', help="Use asimov data instead. Will look for files using [data-asimov] as the naming key instead of [data]. Create asimov data easily using make_asimov.py")
    parser.add_argument('--run-plu-val', action='store_true', help="Runs a PLU validation test by varying the data histogram and performing the entire PLU fit multiple times.")
    parser.add_argument('--condor', action='store_true', help="Runs the GPR fits via HT Condor. Merge the results using merge_gpr_condor.py, then recall master.py using --skip-gpr.")
    parser.add_argument('--mu-stop', default='1,0.05', help="The stop signal strength. Should be a comma-separated pair val,err.")
    parser.add_argument('--channels', default='0,1,2', help="The lepton channels to run over, separated by commas.")
    parser.add_argument('--variables', default='vv_m-mT', help="A tag (see ChannelConfig) that sets the variables to run over.")
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
            utils.Sample.cw_lin,
            utils.Sample.cw_quad,
        ],
        file_path_formats=filepaths,
    )
    return file_manager


def main():
    ### Args ###
    args = parse_args()
    if args.asimov:
        utils.Sample.data.file_stubs = ['data-asimov']
        if args.output.endswith('/'):
            args.output = args.output[:-1]
        args.output += '-asimov'
    if args.no_systs or args.nominal:
        utils.variations_hist.clear()
    mu_stop = [float(x) for x in args.mu_stop.split(',')]
    channels = [int(x) for x in args.channels.split(',')]

    ### Input files ###
    file_manager = get_files(args.filepaths)

    ### Output files ###
    os.makedirs(f'{args.output}/plots', exist_ok=True)
    plot.save_transparent_png = False
    plot.file_formats = ['png', 'pdf']

    ### ttbar fitter ###
    ttbar_fitter = ttbar_fit.TtbarSysFitter(file_manager, mu_stop_0=mu_stop)

    ### Global options ###
    global_config = GlobalConfig(
        ### Basic config ###
        file_manager=file_manager,
        ttbar_fitter=ttbar_fitter,
        mu_stop=mu_stop,
        output_dir=args.output,
        ### Run management ###
        nominal_only=args.nominal,
        skip_hist_gen=args.skip_hist_gen,
        skip_fits=args.skip_fits,
        skip_direct_fit=not args.fit_bin_yields,
        skip_gpr=args.skip_gpr,
        skip_gpr_if_present=not args.rerun_gpr,
        skip_plu=args.skip_plu,
        skip_diboson=args.skip_diboson,
        skip_eft=args.skip_eft,
        gpr_condor=args.condor,
        is_asimov=args.asimov,
        run_plu_val=args.run_plu_val,
    )

    ### Channel options ###
    config = MultiChannelConfig(
        global_config=global_config,
        lepton_channels=channels,
        variable_tag=args.variables,
    )

    ### Single-channel processing ###
    if not args.skip_channels:
        for sc_config in config.split_config():
            run_single_channel(sc_config)

    if len(channels) > 1:
        ### Diboson signal strength fit ###
        run_diboson_fit(config, skip_fits=global_config.skip_fits)

        ### EFT fits ###
        run_eft_fit(config, 'cw_lin', skip_fits=global_config.skip_eft)
        run_eft_fit(config, 'cw_quad', skip_fits=global_config.skip_eft)


if __name__ == "__main__":
    main()
