import ROOT
import numpy as np
import unfolding_rewrite
from plotting import plot
import utils
import sys

# varibales tp set 
lep = 2
lepton_channel = 2 # To  get gpr_asimov for 
sample = utils.Sample.parse("diboson")
var= utils.Variable.vv_m
input_dir_rebin_plots = "2lep_gpr-asimov/rebin"
input_dir_vjets = "2lep_gpr-asimov/gpr"
#
diboson = ROOT.TFile.Open(f"{input_dir_rebin_plots}/{lep}lep_diboson_rebin.root")
vjets = ROOT.TFile.Open(f"{input_dir_vjets}/gpr_{lep}lep_vjets_yield.root")
stop = ROOT.TFile.Open(f"{input_dir_rebin_plots}/{lep}lep_stop_rebin.root")
ttbar = ROOT.TFile.Open(f"{input_dir_rebin_plots}/{lep}lep_ttbar_rebin.root")
h_diboson = diboson.Get(f"diboson_VV{lep}Lep_MergHP_Inclusive_SR_llJ_m")
h_vjets = vjets.Get(f"Vjets_SR_vv_m")
h_stop = stop.Get(f"stop_VV{lep}Lep_MergHP_Inclusive_SR_llJ_m")
h_ttbar = ttbar.Get(f"ttbar_VV{lep}Lep_MergHP_Inclusive_SR_llJ_m")
h_data = h_diboson.Clone()
h_data.Add(h_vjets,1.0)
h_data.Add(h_stop,1.0)
h_data.Add(h_ttbar,1.0)
h_data.SetName(f"data_VV{lep}Lep_MergHP_Inclusive_SR_llJ_m")
output = ROOT.TFile(f"histgpr_data-asimov_{lep}lep-0.root", "RECREATE")
output.cd()
h_data.Write()
output.Close()

