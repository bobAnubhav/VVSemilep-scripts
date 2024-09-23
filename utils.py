'''
@file variable.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date February 7, 2024
@brief Variable naming and other utilities
'''
from __future__ import annotations
from typing import Union
import ctypes
import numpy as np
import os

from plotting import plot
import ROOT # type: ignore

#########################################################################################
###                                     Variables                                     ###
#########################################################################################
class Variable:
    '''
    @property name
        The base name that uniquely identifies the variable. This should match the
        histogram naming conventions.
    @property title
        A TLatex version of the variable that is used in title text.
    @property unit
        The name of the unit of the variable. Again this should match the histogram
        outputs.
    @property rebin
        A default fine rebinning option for diagnostic plots. See [plot.rebin].
    '''
    def __init__(self, name, title, unit, rebin=None, **plot_opts):
        '''
        @param plot_opts
            Any additional default opts to use during plotting, assuming an event
            distribution is being plotted against this variable on the x axis. These can
            be retrieved with [x_plot_opts].
        '''
        self.name = name
        self.title = title
        self.unit = unit
        self.rebin = rebin
        self._plot_opts = plot_opts

    def __repr__(self) -> str:
        return f'Variable({self.name})'
    
    def __str__(self):
        return self.name

    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            if self.unit:
                return f'{self.title} [{self.unit}]'
            else:
                return f'{self.title}'
        else:
            return self.name

    def x_plot_opts(self):
        return {
            'xtitle': f'{self.title} [{self.unit}]',
            **self._plot_opts
        }


Variable.vv_m = Variable(name="vv_m", title="m(VV)", unit="GeV", rebin=100, logy=True)
Variable.vv_mt = Variable(name="vv_mt", title="m_{T}(VV)", unit="GeV", rebin=100, logy=True)

Variable.llJ_m = Variable(name="llJ_m", title="m(llJ)", unit="GeV", rebin=100, logy=True)
Variable.lvJ_m = Variable(name="lvJ_m", title="m(l#nuJ)", unit="GeV", rebin=100, logy=True)
Variable.vvJ_m = Variable(name="vvJ_m", title="m(#nu#nuJ)", unit="GeV", rebin=100, logy=True)

Variable.llJ_mt = Variable(name="llJ_mT", title="m_{T}(llJ)", unit="GeV", rebin=100, logy=True)
Variable.lvJ_mt = Variable(name="lvJ_mT", title="m_{T}(l#nuJ)", unit="GeV", rebin=100, logy=True)
Variable.vvJ_mt = Variable(name="vvJ_mT", title="m_{T}(#nu#nuJ)", unit="GeV", rebin=100, logy=True)

Variable.fatjet_pt = Variable(name="fatjet_pt", title="p_{T}(J)", unit="GeV", rebin=100, logy=True)
Variable.fatjet_m = Variable(name="fatjet_m", title="m(J)", unit="GeV", rebin=2, logy=True)


def generic_var_to_lep(var, lep):
    if var.name == 'vv_m':
        if lep == 0: return Variable.vvJ_m
        if lep == 1: return Variable.lvJ_m
        if lep == 2: return Variable.llJ_m
    elif var.name == 'vv_mt':
        if lep == 0: return Variable.vvJ_mt
        if lep == 1: return Variable.lvJ_mt
        if lep == 2: return Variable.llJ_mt
    return var


#########################################################################################
###                                      Samples                                      ###
#########################################################################################
class Sample:
    '''
    A class that represents a single sample.

    @property name
        A unique name that identifies the sample. This is used to save output files, etc.
    @property title
        A TLatex version of the name that is used in title text.
    @property file_stubs
        A list of the possible naming stubs used in the histogram files for this sample.
    @property hist_keys
        A list of the naming stubs used in the histogram names for this sample.
    '''
    ### Static members ###
    # List here for linting, initialized later below
    wjets : Sample = None
    zjets : Sample = None
    vjets : Sample = None
    ttbar : Sample = None
    stop : Sample = None
    diboson : Sample = None
    data : Sample =  None
    cw_lin : Sample = None
    cw_quad : Sample = None

    def __init__(self, name : str, title : str, file_stubs : list[str], hist_keys : list[str]):
        self.name = name
        self.title = title
        self.file_stubs = file_stubs
        self.hist_keys = hist_keys

    def __format__(self, __format_spec: str) -> str:
        if __format_spec:
            return f'{self.title}'
        else:
            return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            raise TypeError(f"Sample.__eq__({type(other)})")

    @staticmethod
    def list_predefined():
        return [Sample.wjets, Sample.zjets, Sample.ttbar, Sample.stop, Sample.diboson, Sample.data]

    @staticmethod
    def parse(name) -> Sample:
        samples = Sample.list_predefined()
        for x in samples:
            if x.name == name:
                return x
        raise RuntimeError(f"Sample.parse() couldn't parse {name}")


Sample.wjets = Sample('wjets', 'W+jets', ['Wjets_Sherpa2211'], ['WLL', 'WHL', 'WHH', 'Wjets'])
Sample.zjets = Sample('zjets', 'Z+jets', ['Zjets_Sherpa2211'], ['ZLL', 'ZHL', 'ZHH', 'Zjets'])
Sample.vjets = Sample('vjets', 'V+jets', ['Wjets_MG','Zjets_MG'], ['ZLL', 'ZHL', 'ZHH', 'Zjets','WLL', 'WHL', 'WHH', 'Wjets'])
Sample.ttbar = Sample('ttbar', 't#bar{t}', ['ttbar'], ['ttbar'])
Sample.stop = Sample('stop', 'single top', ['stop'], ['stop'])
Sample.diboson = Sample('diboson', 'diboson', ['diboson_Sherpa2211', 'Diboson_Sh2211'], ['SMVV', 'WW', 'WZ', 'ZZ'])
Sample.data = Sample('data', 'data', ['data', 'data15', 'data16', 'data17', 'data18'], ['data'])

Sample.cw_lin = Sample('cw_lin', 'c_{W}', ['EFT'], [f'VV{x}qq_NPSQeq1_cW_1' for x in ['vv', 'lv', 'll']])
Sample.cw_quad = Sample('cw_quad', 'c_{W}^{2}', ['EFT'], [f'VV{x}qq_NPeq1_cW_1' for x in ['vv', 'lv', 'll']])


#########################################################################################
###                                      Binning                                      ###
#########################################################################################


def get_bins(lepton_channel : int, var: Variable):
    if var.name in ["vv_m", "lvJ_m", "llJ_m"]:
        if lepton_channel == 0:
            # below, but matched to 1lep custom since they're pretty similar
            return [700, 810, 940, 1090, 1260, 1500, 2000, 6000]
            # optimized binning with threshold_diag=0.7, threshold_err=0.2, min_reco_count=10
            return [700, 790, 930, 1090, 1260, 1440, 1650, 1880, 3000]
        elif lepton_channel == 1:
            # below, but with custom fixes
            return [700, 810, 940, 1090, 1260, 1500, 2000, 6000]
            # optimized binning with threshold_diag=0.6, threshold_err=0.4, monotonic_bin_sizes=False,
            return [500, 690, 810, 940, 1090, 1260, 1450, 1640, 1850, 2090, 2390, 3000]
            # optimized binning with threshold_diag=0.7, threshold_err=0.2, min_reco_count=10
            return [700, 790, 920, 1070, 1240, 1420, 1610, 1820, 3000]
            # old binning
            return [500, 600, 700, 800, 900, 1020, 1170, 1310, 1470, 1780, 2090, 2400, 3000]
        elif lepton_channel == 2:
            # optimized binning with threshold_diag=0.8, threshold_err=0.2, min_reco_count=10
            return [700, 770, 890, 1040, 1210, 1400, 1640, 6000]
            # manually adjusted
            return [700, 750, 800, 900, 1000, 1100, 1250, 1400, 3000]
            # optimized binning with threshold_diag=0.7, threshold_err=0.2, min_reco_count=10
            return [700, 750, 830, 910, 1010, 1110, 1220, 1380, 3000]
    elif var.name == "vv_mt" or var.name == "vvJ_mT":
        if lepton_channel == 0:
            # optimized binning with threshold_diag=0.7, threshold_err=0.2, min_reco_count=10
            return [600, 690, 810, 940, 1090, 1260, 1430, 1630, 1850, 6000]
        elif lepton_channel == 1:
            # optimized binning with threshold_diag=0.7, threshold_err=0.2, min_reco_count=10
            return [600, 710, 830, 960, 1100, 1260, 1430, 1600, 6000]
    elif var.name == "fatjet_pt" or var.name == "vhad_pt":
        if lepton_channel == 0:
            # optimized binning with threshold_diag=0.8, threshold_err=0.2, min_reco_count=10
            return [300, 360, 460, 580, 730, 3000]
        elif lepton_channel == 1:
            # optimized binning with threshold_diag=0.8, threshold_err=0.2, min_reco_count=10
            return [300, 360, 460, 580, 730, 3000]
        elif lepton_channel == 2:
            # optimized binning with threshold_diag=0.8, threshold_err=0.2, min_reco_count=10
            return [300, 350, 450, 570, 3000]

    raise NotImplementedError(f"utils.py::get_bins({lepton_channel}, {var.name})")

def get_adjusted_bins(lepton_channel: int, var: Variable):
    '''
    The large overflow bin looks bad in distribution plots. So use these adjusted bins
    instead, but just for plotting.
    '''
    bins = get_bins(lepton_channel, var)
    if var in [Variable.vv_m, Variable.vv_mt]: 
        bins[-1] = 3000
    elif var in [Variable.fatjet_pt]:
        bins[-1] = 1500
    return np.array(bins, dtype=float)


#########################################################################################
###                                    Variations                                     ###
#########################################################################################

variation_up_key = '__1up'
variation_down_key = '__1down'

variation_nom = 'nominal'
variation_lumi = 'lumiNP'

variation_mu_ttbar = 'mu-ttbar'
variation_mu_stop = 'mu-stop'
variations_custom = [
    variation_mu_ttbar,
    variation_mu_stop,
]

variations_hist = [
    'SysEG_RESOLUTION_ALL',
    'SysEG_SCALE_ALL',
    'SysEL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR',
    'SysEL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR',
    'SysEL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR',
    'SysEL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR',
    'SysEL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR',
    'SysFT_EFF_Eigen_B_0_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_0_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_1_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_1_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_2_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_2_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_3_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_3_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_4_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_4_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_5_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_5_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_6_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_6_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_B_7_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_B_8_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_C_0_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_C_0_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_C_1_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_C_1_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_C_2_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_C_2_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_C_3_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_C_3_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_0_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_0_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_1_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_1_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_2_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_2_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_3_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_3_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_4_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_5_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_Eigen_Light_6_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_extrapolation_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_extrapolation_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysFT_EFF_extrapolation_from_charm_AntiKt4EMPFlowJets_BTagging201903',
    'SysFT_EFF_extrapolation_from_charm_AntiKtVR30Rmax4Rmin02TrackJets_BTagging201903',
    'SysJET_BJES_Response',
    'SysJET_EffectiveNP_Detector1',
    'SysJET_EffectiveNP_Detector2',
    'SysJET_EffectiveNP_Mixed1',
    'SysJET_EffectiveNP_Mixed2',
    'SysJET_EffectiveNP_Mixed3',
    'SysJET_EffectiveNP_Modelling1',
    'SysJET_EffectiveNP_Modelling2',
    'SysJET_EffectiveNP_Modelling3',
    'SysJET_EffectiveNP_Modelling4',
    'SysJET_EffectiveNP_R10_Detector1',
    'SysJET_EffectiveNP_R10_Detector2',
    'SysJET_EffectiveNP_R10_Mixed1',
    'SysJET_EffectiveNP_R10_Mixed2',
    'SysJET_EffectiveNP_R10_Mixed3',
    'SysJET_EffectiveNP_R10_Mixed4',
    'SysJET_EffectiveNP_R10_Modelling1',
    'SysJET_EffectiveNP_R10_Modelling2',
    'SysJET_EffectiveNP_R10_Modelling3',
    'SysJET_EffectiveNP_R10_Modelling4',
    'SysJET_EffectiveNP_R10_Modelling5',
    'SysJET_EffectiveNP_R10_Modelling6',
    'SysJET_EffectiveNP_R10_Modelling7',
    'SysJET_EffectiveNP_R10_Statistical1',
    'SysJET_EffectiveNP_R10_Statistical2',
    'SysJET_EffectiveNP_R10_Statistical3',
    'SysJET_EffectiveNP_R10_Statistical4',
    'SysJET_EffectiveNP_R10_Statistical5',
    'SysJET_EffectiveNP_R10_Statistical6',
    'SysJET_EffectiveNP_R10_Statistical7',
    'SysJET_EffectiveNP_R10_Statistical8',
    'SysJET_EffectiveNP_Statistical1',
    'SysJET_EffectiveNP_Statistical2',
    'SysJET_EffectiveNP_Statistical3',
    'SysJET_EffectiveNP_Statistical4',
    'SysJET_EffectiveNP_Statistical5',
    'SysJET_EffectiveNP_Statistical6',
    'SysJET_EtaIntercalibration_Modelling',
    'SysJET_EtaIntercalibration_R10_TotalStat',
    'SysJET_EtaIntercalibration_TotalStat',
    'SysJET_Flavor_Composition',
    'SysJET_Flavor_Response',
    #'SysJET_JERMC_EffectiveNP_1', # Missing __1down histos in all files
    'SysJET_JERMC_EffectiveNP_10',
    'SysJET_JERMC_EffectiveNP_11',
    #'SysJET_JERMC_EffectiveNP_12restTerm', # Missing __1down histos in all files
    'SysJET_JERMC_EffectiveNP_2',
    'SysJET_JERMC_EffectiveNP_3',
    'SysJET_JERMC_EffectiveNP_4',
    'SysJET_JERMC_EffectiveNP_5',
    'SysJET_JERMC_EffectiveNP_6',
    'SysJET_JERMC_EffectiveNP_7',
    'SysJET_JERMC_EffectiveNP_8',
    'SysJET_JERMC_EffectiveNP_9',
    #'SysJET_JERPD_DataVsMC_MC16', # Missing __1down histos in all files
    'SysJET_JERPD_EffectiveNP_10',
    'SysJET_JERPD_EffectiveNP_11',
    'SysJET_JERPD_EffectiveNP_2',
    'SysJET_JERPD_EffectiveNP_3',
    'SysJET_JERPD_EffectiveNP_4',
    'SysJET_JERPD_EffectiveNP_5',
    'SysJET_JERPD_EffectiveNP_6',
    'SysJET_JERPD_EffectiveNP_7',
    'SysJET_JERPD_EffectiveNP_8',
    'SysJET_JERPD_EffectiveNP_9',
    'SysJET_JMS_Topology_QCD',
    'SysJET_JvtEfficiency',
    'SysJET_LargeR_TopologyUncertainty_V',
    'SysJET_LargeR_TopologyUncertainty_top',
    'SysJET_Pileup_OffsetMu',
    'SysJET_Pileup_OffsetNPV',
    'SysJET_Pileup_PtTerm',
    'SysJET_Pileup_RhoTopology',
    'SysJET_Rtrk_Baseline_frozen_mass',
    'SysJET_Rtrk_ExtraComp_Baseline_frozen_mass',
    'SysJET_Rtrk_ExtraComp_Modelling_frozen_mass',
    'SysJET_Rtrk_Modelling_frozen_mass',
    #'SysMET_SoftTrk_ResoPara', # Missing __1down histos in all files
    #'SysMET_SoftTrk_ResoPerp', # Missing __1down histos in all files
    'SysMET_SoftTrk_Scale',
    'SysMUON_EFF_ISO_STAT',
    'SysMUON_EFF_ISO_SYS',
    'SysMUON_EFF_RECO_STAT',
    'SysMUON_EFF_RECO_STAT_LOWPT',
    'SysMUON_EFF_RECO_SYS',
    'SysMUON_EFF_RECO_SYS_LOWPT',
    'SysMUON_EFF_TTVA_STAT',
    'SysMUON_EFF_TTVA_SYS',
    'SysMUON_EFF_TrigStatUncertainty',
    'SysMUON_EFF_TrigSystUncertainty',
    'SysMUON_ID',
    'SysMUON_MS',
    'SysMUON_SAGITTA_RESBIAS',
    'SysMUON_SCALE',
    'SysPRW_DATASF',
    'SysTAUS_TRUEELECTRON_EFF_ELEBDT_STAT',
    'SysTAUS_TRUEELECTRON_EFF_ELEBDT_SYST',
    'SysTAUS_TRUEHADTAU_EFF_ELEOLR_TOTAL',
    'SysTAUS_TRUEHADTAU_EFF_RECO_TOTAL',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_1PRONGSTATSYSTPT2025',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_1PRONGSTATSYSTPT2530',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_1PRONGSTATSYSTPT3040',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_1PRONGSTATSYSTPTGE40',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_3PRONGSTATSYSTPT2025',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_3PRONGSTATSYSTPT2530',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_3PRONGSTATSYSTPT3040',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_3PRONGSTATSYSTPTGE40',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_HIGHPT',
    'SysTAUS_TRUEHADTAU_EFF_RNNID_SYST',
    'SysTAUS_TRUEHADTAU_SME_TES_DETECTOR',
    'SysTAUS_TRUEHADTAU_SME_TES_INSITUEXP',
    'SysTAUS_TRUEHADTAU_SME_TES_INSITUFIT',
    'SysTAUS_TRUEHADTAU_SME_TES_MODEL_CLOSURE',
    'SysTAUS_TRUEHADTAU_SME_TES_PHYSICSLIST',
]

Variation_hist_onesided = [
    'SysJET_JERMC_EffectiveNP_1',
    'SysJET_JERMC_EffectiveNP_12restTerm',
    'SysJET_JERPD_DataVsMC_MC16',
    'SysMET_SoftTrk_ResoPara',
    'SysMET_SoftTrk_ResoPerp',
]

def is_histo_syst(x):
    if variation_nom in x: return False
    if variation_lumi in x: return False
    if 'mu-diboson' in x: return False
    for var in variations_custom:
        if var in x: return False
    return True


def hist_name_variation(hist_name, sample : Sample, variation, separator='_'):
    if not is_histo_syst(variation):
        return hist_name
    if sample == Sample.data:
        return hist_name
    return hist_name + separator + variation


#########################################################################################
###                                    File Access                                    ###
#########################################################################################

def get_hist(tfile, hist_name):
    '''
    Retrieves [hist_name] from [tfile] with error checking.
    '''
    h = tfile.Get(hist_name)
    if not h or h.ClassName() == 'TObject':
        raise RuntimeError(f"Couldn't retrieve histogram {hist_name} from {tfile}")
    return h


def get_hists_sum(tfile, hist_names):
    '''
    Retrieves the histograms specified in [hist_names] from [tfile] and returns their sum.
    Will skip missing histograms.
    '''
    h_sum = None
    for name in hist_names:
        h2 = tfile.Get(name)
        if not h2 or h2.ClassName() == 'TObject':
            plot.warning(f"Couldn't retrieve histogram {name} from {tfile}.")
        elif h_sum is None:
            h_sum = h2.Clone()
        else:
            h_sum.Add(h2)
    if h_sum is None:
        raise RuntimeError(f"get_hists_sum() unable to retrieve histograms from {tfile}.")
    return h_sum


class FileManager:
    '''
    This class helps manage retrieving histograms for many samples from a multitude of
    files. In general, each sample may be split across multiple files and the histograms
    divided by subsamples. This class will take care of adding them all together.

    Make sure though that there are not redundant files that accidentally duplicate the
    data!
    '''

    _filesystem_case_insensitive = os.path.exists(__file__.upper()) and os.path.exists(__file__.lower())

    lepton_channel_names = {
        0: ['0lep', '0Lep', 'HIGG5D1'], # DO NOT MODIFY without checking the _filesystem_case_insensitive test in _get_sample_files
        1: ['1lep', '1Lep', 'HIGG5D2'],
        2: ['2lep', '2Lep', 'HIGG2D4'],
    }

    def __init__(self, samples : list[Sample], file_path_formats : list[str], lepton_channels = [0, 1, 2]) -> None:
        '''
        Tries to open every file given a set of samples and path formats.

        @param file_path_formats
            Pass any number file paths that can optionally contain fields "{lep}" which
            will be replaced with each channel in [lepton_channels] using
            [lepton_channel_names] and "{sample}" which will be replaced with
            [Sample.file_stubs] for each sample.
        '''
        self.samples = { sample.name : sample for sample in samples }
        self.file_path_formats = file_path_formats
        self.files = {
            (lep, sample.name) : self._get_sample_files(lep, sample, file_path_formats)
            for lep in lepton_channels
            for sample in samples
        }

        self._log_cache : set[tuple[int, str, str]] = set() # (lep, sample.name, file_format) to supress log messages

    def _get_sample_files(self, lep : int, sample : Sample, file_path_formats: list[str]) -> list[ROOT.TFile]:
        files = []

        for path in file_path_formats:
            sample_keys = sample.file_stubs if '{sample}' in path else [None]
            lep_keys = self.lepton_channel_names[lep] if '{lep}' in path else [None]
            for sample_stub in sample_keys:
                for lep_stub in lep_keys:
                    if self._filesystem_case_insensitive and lep_stub is not None and lep_stub.endswith('Lep'): 
                        # DON'T try both 1lep and 1Lep in case-insensitive file systems or
                        # else the file will be duplicated
                        continue 
                    formatted_path = path.format(lep=lep_stub, sample=sample_stub)
                    ROOT.gSystem.RedirectOutput("/dev/null") # mute TFile error messages
                    try:
                        files.append(ROOT.TFile(formatted_path))
                    except OSError as e:
                        pass
                    ROOT.gSystem.RedirectOutput(plot.nullptr_char, plot.nullptr_char)

        if not files:
            plot.warning(f'FileManager() unable to find files for {sample} in the {lep}-lep channel.')
        elif len(files) > 1:
            plot.warning(f'FileManager() multiple files found for {sample} in the {lep}-lep channel. These will be added together!!!')
        return files

    def get_hist(self, lep : int, sample : Union[str, Sample], hist_name_format : str, variation : str = variation_nom) -> Union[ROOT.TH1F, None]:
        '''
        Retrieves a single histogram given a sample and name format. Histograms from every
        sample file in [self.files] using every key in [sample.hist_keys] will be fetched
        and added together.

        @param hist_name_format
            The name of the histogram, which can optionally contain formatters "{lep}"
            which will be replaced with [lep] and "{sample}" which will be replaced with
            [sample.hist_keys].
        '''
        if isinstance(sample, str):
            sample = self.samples[sample]
        files = self.files[(lep, sample.name)]

        h_out = None
        sample_keys = sample.hist_keys if '{sample}' in hist_name_format else [None]
        lep_keys = self.lepton_channel_names[lep] if '{lep}' in hist_name_format else [None]
        used_keys = [] # (sample, lep, file)
        for key in sample_keys:
            for lep_name in lep_keys:
                name = hist_name_format.format(lep=lep_name, sample=key)
                name = hist_name_variation(name, sample, variation)
                for file in files:
                    h = file.Get(name)
                    if not h or h.ClassName() == 'TObject':
                        continue
                    elif h_out is None:
                        used_keys.append((key, lep_name, file.GetName()))
                        h_out = h.Clone()
                    else:
                        used_keys.append((key, lep_name, file.GetName()))
                        h_out.Add(h)

        if h_out is None:
            if variation != variation_nom:
                plot.warning(f'FileManager() unable to find histogram {hist_name_format} with variation {variation} for {sample} in the {lep}-lep channel.')
            else:
                plot.warning(f'FileManager() unable to find histogram {hist_name_format} for {sample} in the {lep}-lep channel.')
        elif len(used_keys) > 1:
            cache_key = (lep, sample.name, hist_name_format)
            if cache_key not in self._log_cache:
                self._log_cache.add(cache_key)
                msg = f"FileManager.get_hist() Added {len(used_keys)} histograms for {hist_name_format}:"
                for skey,lkey,fname in used_keys:
                    msg += f'\n    sample={skey}, lep={lkey}, file={fname}'
                plot.warning(msg)
        return h_out

    def get_file_names(self, lep : int, sample : Union[str, Sample]) -> list[str]:
        '''
        Returns the list of files matching this (lep, sample).
        '''
        if isinstance(sample, str):
            sample = self.samples[sample]
        files = self.files[(lep, sample.name)]
        return [file.GetName() for file in files]

    def get_hist_all_samples(self, lep : int, hist_name_format : str, variation : str = variation_nom) -> dict[str, ROOT.TH1F]:
        '''
        Like [get_hist] but returns a dict of all the histograms for each sample.
        '''
        out = {}
        for _,sample in self.samples.items():
            out[sample.name] = self.get_hist(lep, sample, hist_name_format, variation)
        return out
