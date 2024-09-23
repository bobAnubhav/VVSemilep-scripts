#!/usr/bin/env python3
'''
@file unfolding.py
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch
@date April 27, 2023
@brief Unfolding postscripts for creating response matrices and ResonanceFinder histograms

This creates the following plots:

    - migration_matrix.png/pdf 
    - eff_acc.png/pdf
    - fid_reco.png/pdf

and the following output file for RF input:

    - rf_histograms.root


------------------------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------------------------

    setupATLAS 
    lsetup "root recommended" 
    lsetup "python centos7-3.9"

Note that this can't be setup at the same time with AnalysisBase or else you get a lot of
conflicts :(

------------------------------------------------------------------------------------------
CONFIG
------------------------------------------------------------------------------------------

Check [utils.Sample] to make sure the hardcoded naming stuctures are correct. Also check
the hardcoded variables and binning in [utils.get_bins]. If optimizing bins, check the
parameters in [optimize_binning].

------------------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------------------

    unfolding.py path/to/hists/{lep}_{sample}-0.root diboson 1

Where the filepath includes formatters. See [utils.FileManager]. Optionally specify

    --optimize 500,3000

To run the automatic bin optimization.
------------------------------------------------------------------------------------------
run with --variation flag to create the unfolding matrix for all variations
------------------------------------------------------------------------------------------
'''

from plotting import plot
import ROOT # type: ignore
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import utils

##############################################################################
###                               UTILITIES                                ###
##############################################################################

def get_migration_matrix(h_reco):
    '''
    Creates a migration matrix from an input 2d distribution of fid vs reco filled with reconstructed weights 

    @param h
        Input 2d histogram of fid vs reco (filled with reconstructed weights). The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
    @return
        A new TH2F with matching bins as [h]. The underflows should not be used anymore.
    '''
    h = h_reco.Clone()

    ### Remove negative bins ###
    tot_preclean = h.ProjectionX('_px', 1, h.GetNbinsY())
    for x in range(1, h.GetNbinsX() + 1):
        for y in range(1, h.GetNbinsY() + 1):
            v = h.GetBinContent(x, y)
            if v < 0:
                plot.warning("get_migration_matrix() Negative bin @ "
                             f"fid [{h.GetXaxis().GetBinLowEdge(x)}, {h.GetXaxis().GetBinLowEdge(x+1)}], " 
                             f"reco [{h.GetYaxis().GetBinLowEdge(y)}, {h.GetYaxis().GetBinLowEdge(y+1)}]: "
                             f"{v} / {tot_preclean.GetBinContent(x)} ({100 * v / tot_preclean.GetBinContent(x):.2f}%)")
                h.SetBinContent(x, y, 0)
    
    ### Normalize each column ###
    tot_fid = h.ProjectionX('_px', 1, h.GetNbinsY()) # The denominator should only be the "visible" bins
    # here the Migration matrix is normailsed by the fiducial bin with recontructed weights 
    for x in range(1, h.GetNbinsX() + 1):
        tot = tot_fid.GetBinContent(x)
        if tot <= 0: continue
        for y in range(1, h.GetNbinsY() + 1):
            v = h.GetBinContent(x, y)
            e = h.GetBinError(x, y)
            h.SetBinContent(x, y, v / tot)
            h.SetBinError(x, y, e / tot) # TODO is this right
    
    return h


def get_unfolding_efficiency(h_reco, h_fid):
    '''
    Creates an efficiency histogram from an input 2d distribution of fid vs reco.

    @param h_reco
        Input 2d histogram of reco vs fid filled with reco weights . The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
        The underflow of the y/x axes should be the mis-efficieny/acceptance respectively, the
        events that fall into only one of either reco/fiducial selections.
    @param h_fid
        Input 2d histogram of reco vs fid filled with fiducial weights. The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.

    @return
        A new TH1F with the bins matching the x axis bins of [h], containing the efficiency values.
    '''
  
    h_fid_mod = h_fid.ProjectionX()
    total = h_fid_mod # total fiducial events in each bin using fiducial weights 
    
    eff = h_reco.ProjectionX(h_reco.GetName() + '_eff', 1, h_reco.GetNbinsY()) # skip underflow bin == miseff
    eff.Divide(total)
    return eff


def get_unfolding_accuracy(h_reco):
    '''
    Creates an acceptance histogram from an input 2d distribution of fid vs reco.

    @param h
        Input 2d histogram of reco vs fid filled with reco wreights . The y axis should be the reco distribution of
        a variable, and the x axis should be the fiducial distrubtion of the same variable.
        The underflow of the y/x axes should be the mis-efficieny/acceptance respectively, the
        events that fall into only one of either reco/fiducial selections.

    @return
        A new TH1F with the bins matching the y axis bins of [h], containing the acceptance values.
    '''
    total = h_reco.ProjectionY() # total reco events in each bin using reco weights
    acc = h_reco.ProjectionY(h_reco.GetName() + '_acc', 1, h_reco.GetNbinsX()) # skip underflow bin == misacc
    acc.Divide(total)
    return acc


def get_response_matrix(h_reco, h_fid):
    '''
    Creates the response matrix from an input 2d distribution of fid vs reco.

    @see [get_migration_matrix]
    '''
    mig_mtx = get_migration_matrix(h_reco)
    efficiency = get_unfolding_efficiency(h_reco, h_fid)
    acceptance = get_unfolding_accuracy(h_reco)

    for y in range(1, mig_mtx.GetNbinsY() + 1):
        for x in range(1, mig_mtx.GetNbinsX() + 1):
            mig = mig_mtx.GetBinContent(x, y)
            err = mig_mtx.GetBinError(x, y)
            eff = efficiency.GetBinContent(x)
            acc = acceptance.GetBinContent(y)
            if acc > 0:
                mig_mtx.SetBinContent(x, y, eff / acc * mig)
                mig_mtx.SetBinError(x, y, eff / acc * err) # TODO is this right?
            else:
                mig_mtx.SetBinContent(x, y, 0)
                mig_mtx.SetBinError(x, y, 0)

    return mig_mtx


def optimize_binning(
        h, fid_range,
        threshold_diag=0.7,
        max_mc_frac_err=0.2,
        min_reco_count=10,
        monotonic_bin_sizes=False,
    ):
    '''
    Attempts to optimize the binning for unfolding such that migration matrix is
    well-behaved. It checks for the following conditions:
        1. The diagonal entries have at least [thresold_diag] fraction of the events.
        2. The fractional MC stat error of the fiducial distribution < [max_mc_frac_err]
           in each bin.
        3. The reco event count in the bin is >= [min_reco_count]
        4. If [montonic_bin_sizes], bins at higher m/pT must be larger than the ones at
           smaller m/pT.

    @param h
        Input 2d histogram of fid vs reco. The y axis should be the reco distribution of a
        variable, and the x axis should be the fiducial distrubtion of the same variable.
    @param fid_range
        The final range that should be covered by the bins.
    '''
    ### Range ###
    h_1d = h.ProjectionX()
    i_start = h_1d.FindFixBin(fid_range[0])
    i_end = h_1d.FindFixBin(fid_range[1])
    i = i_start

    ### Denominator in truth bin ###
    h_reco = h.ProjectionY()
    h_fid = h.ProjectionX('_px', i_start, i_end - 1) # end is inclusive :)

    ### Running bins ###
    bin_edges = [fid_range[0]]
    bin_indices = [i] # unused...
    last_merge_size = 0

    ### Loop once per NEW (merged) bin ###
    while i < i_end:
        n_tot = 0
        n_diag = 0
        n_reco = 0
        err_tot = 0
        merge_size = 0
        
        ### Each loop here adds one fine bin into the merged bin ###
        while (n_diag <= 0
               or n_diag / n_tot < threshold_diag
               or (monotonic_bin_sizes and merge_size < last_merge_size)
               or err_tot**0.5 / n_tot > max_mc_frac_err
               or n_reco < min_reco_count
            ):
            
            ### Last bin incomplete ###
            if i + merge_size >= i_end:
                # Merge it with the previously completed bin by popping the last edge
                bin_edges.pop()
                bin_indices.pop()
                break

            ### Increase bin size by 1 fine bin ###
            n_reco += h_reco.GetBinContent(i + merge_size)
            n_tot += h_fid.GetBinContent(i + merge_size)
            err_tot += h_fid.GetBinError(i + merge_size)**2

            ### Merge diagonal ###
            # Merge the diagonal count by essentially adding "L" blocks 
            # First add top of "L"
            for x in range(i, i + merge_size):
                n_diag += h.GetBinContent(x, i + merge_size)
            # Then add right of "L"
            for y in range(i, i + merge_size + 1):
                n_diag += h.GetBinContent(i + merge_size, y)
            merge_size += 1

        ### Add end of this bin ###
        i += merge_size
        bin_edges.append(h.GetXaxis().GetBinLowEdge(i))
        bin_indices.append(i)
        last_merge_size = merge_size

    plot.success(f'unfolding.py Optimized bins: {h.GetName()} {[int(x) for x in bin_edges]}')
    return bin_edges


def output_path(output_dir, sample, lepton_channel=None):
    out = f'{output_dir}/{sample}_{{lep}}lep_rf_histograms.root'
    if lepton_channel is not None:
        return out.format(lep=lepton_channel)
    else:
        return out


##############################################################################
###                                PLOTTING                                ###
##############################################################################

def plot_migration_matrix(mtx_reco, var, **kwargs):
    '''
    Plots the 2D migration matrix in COLZ mode with overlaid text.
    '''
    ROOT.gStyle.SetPaintTextFormat('2.0f')
    ROOT.gStyle.SetPalette(ROOT.kAlpine)

    ### For plotting, hide small bins. Otherwise colz shows a sea of blue ###
    h = get_migration_matrix(mtx_reco)
    for y in range(1, h.GetNbinsY() + 1):
        for x in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(x, y)
            if v < 0.01:
                h.SetBinContent(x, y, 0)
            else:
                h.SetBinContent(x, y, v * 100)

    ### Draw ###
    plot.plot([h],
        ytitle=f'Detector {var:title}',
        xtitle=f'Fiducial {var:title}',
        opts='COLZ TEXT',
        **kwargs,
    )
    ROOT.gStyle.SetPalette(ROOT.kBird)


def plot_matrix_tiered(h_reco, var, **kwargs):
    '''
    Plots the "signal histogram" of each fiducial bin
    '''
    projections = []
    labels = []
    h = h_reco.Clone()
    for i in range(1, h.GetNbinsX() + 1): 
        projections.append([h.ProjectionY(f'p{i}', i, i)])
        labels.append(f'{int(h.GetXaxis().GetBinLowEdge(i))},{int(h.GetXaxis().GetBinLowEdge(i+1))}')

    plot.plot_tiered(
        projections,
        tier_labels=labels,
        tier_title=f'Fiducial {var:title}',
        ytitle='Normalized Events',
        xtitle=f'Detector {var:title}',
        plot_style='fill',
        fillcolor=plot.colors.tableu_40,
        y_pad_top=0.2,
        **kwargs,
    )


def plot_eff_acc(mtx_reco,mtx_fid, var, **kwargs):
    efficiency = get_unfolding_efficiency(mtx_reco,mtx_fid)
    accuracy = get_unfolding_accuracy(mtx_reco)
    plot.plot_2panel([efficiency], [accuracy],
        xtitle=f'{var:title}',
        ytitle='Efficiency',
        ytitle2='#splitline{Fiducial}{Accuracy}',
        # y_range=[0, 0.5],
        # y_range2=[0.5, 1],
        **kwargs
    )


def plot_fid_reco(mtx_reco, mtx_fid, var, **kwargs):
    
    fid = mtx_fid.ProjectionX() 
    reco = mtx_reco.ProjectionY() # reco distribution
    int_reco = mtx_reco.ProjectionY('int_reco', 1, mtx_reco.GetNbinsY())

    fid.Scale(1, 'width')
    reco.Scale(1, 'width')
    int_reco.Scale(1, 'width')

    ratio = fid.Clone()
    ratio.Divide(reco)

    plot.plot_ratio([fid, reco], [ratio],
        legend=['Fiducial', 'Detector'],
        ytitle='Events / Bin Width',
        ytitle2='#frac{Fiducial}{Detector}',
        xtitle=f'{var:title}',
        linecolor=plot.colors.tableu,
        markercolor=plot.colors.tableu,
        logy=True,
        **kwargs,
    )




##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

_default_vars = [
    # utils.Variable.fatjet_pt,
    utils.Variable.vv_m,
    # utils.Variable.vv_mt,
]

def create_response(
        file_manager : utils.FileManager,
        sample : utils.Sample, 
        lepton_channel : int, 
        rf_output_file : ROOT.TFile,
        rf_output_path : str,
        output_plots : str,
        vars : list[utils.Variable] = _default_vars,
        optimization_range : tuple[float, float] = None,
        variation = False,
        variation_name: str ="",
    ):
    '''
    Runs the full script for a single [sample] and [lepton_channel] but possible many [vars]. 
    See file docstring for more details.

    @param file_manager: An instance of the FileManager class.
    @param sample: The sample to be processed.
    @param lepton_channel: The lepton channel.
    @param rf_output_file: The ROOT TFile object to write the output to.
    @param rf_output_path: The path to the output file.
    @param output_plots: The base directory for all output plots.
    @param vars: A list of Variable objects representing the variables to be processed.
    @param optimization_range: A tuple specifying the range for auto-optimizing the binning.
    @param variation: A boolean indicating whether variation is enabled.
    @param variation_name: The name of the variation.

    @return: The path to the output file.
    '''

    ### Config ###
    common_subtitle = [
        '#sqrt{s}=13 TeV, 140 fb^{-1}',
        f'{lepton_channel}-lepton channel, {sample}',
    ]
  

    ### Run ###
    for var in vars:
        ### Get base histogram ###
        if variation:
            mtx_reco = file_manager.get_hist(
                    lep=lepton_channel, 
                    sample=sample.name, 
                    hist_name_format= '{sample}_VV{lep}_Merg_unfoldingMtx_' + f'{var}',
                    variation = variation_name
                )
        else:
            mtx_reco = file_manager.get_hist(
            lep=lepton_channel, 
            sample=sample.name, 
            hist_name_format= '{sample}_VV{lep}_Merg_unfoldingMtx_' + f'{var}'
        )

        temp_file_manager = utils.FileManager(
        samples=[utils.Sample.diboson],
        file_path_formats=['/eos/user/a/anubhav/phd/cxAOD_out_grid/hist_28July/{lep}_diboson_hist.root'],
        lepton_channels=[0, 1, 2],
        )
        mtx_fid = temp_file_manager.get_hist(
            lep=lepton_channel, 
            sample=sample.name, 
            hist_name_format='{sample}_VV{lep}_Merg_unfoldingMtx_' + f'{var}' + '_fidWeight'
        )
        
        # mtx_fid = file_manager.get_hist(
        #     lep=lepton_channel, 
        #     sample=sample.name, 
        #     hist_name_format='{sample}_VV{lep}_Merg_unfoldingMtx_' + f'{var}' #+'_fidWeight'
        # )
        
        ### Rebin ###
        if optimization_range:
            bins = optimize_binning(mtx_reco, optimization_range)
        else:
            bins = utils.get_bins(lepton_channel, var)
        mtx_reco = plot.rebin2d(mtx_reco, bins, bins)
        mtx_fid = plot.rebin2d(mtx_fid, bins, bins)
        reponse_matrix = get_response_matrix(mtx_reco,mtx_fid)
    
        ### Plot ###
        if variation:
            output_plot_basepath = f'{output_plots}/{lepton_channel}lep_{var}.{sample}_{variation_name}'
        else:
            output_plot_basepath = f'{output_plots}/{lepton_channel}lep_{var}.{sample}'
        plot_migration_matrix(mtx_reco, var,
            filename=f'{output_plot_basepath}_migration_matrix',
            subtitle=[
                *common_subtitle,
                '% migration from each fiducial bin'
            ],
        )
        plot_matrix_tiered(get_migration_matrix(mtx_reco), var,
            filename=f'{output_plot_basepath}_migration_matrix_tiered',
            subtitle=common_subtitle,
        )
        plot_matrix_tiered(reponse_matrix, var,
            filename=f'{output_plot_basepath}_response_matrix_tiered',
            subtitle=common_subtitle,
        )
        plot_eff_acc(mtx_reco,mtx_fid, var,
            filename=f'{output_plot_basepath}_eff_acc',
            subtitle=common_subtitle,
        )
        plot_fid_reco(mtx_reco,mtx_fid, var,
            filename=f'{output_plot_basepath}_fid_reco',
            subtitle=common_subtitle,
        )
        
        rf_output_file.cd() #anubhav
        ### Save ###
        for x in range(1, reponse_matrix.GetNbinsX() + 1):
            if variation:
                name = f'ResponseMatrix_{var}_fid' + str(x).rjust(2, '0') + '__' + variation_name
            else:
                name = f'ResponseMatrix_{var}_fid' + str(x).rjust(2, '0')
            p = reponse_matrix.ProjectionY(name, x, x)
            # Need to convert TH1D to TH1F for ResonanceFinder!!!! Or else it death spirals :)
            h = ROOT.TH1F(f'temp_{var}_{x}', '', 1, 0, 1)
            p.Copy(h) # Yes, this is the correct syntax. Physicists are awesome.
            h.Write(plot.nullptr_char, ROOT.TObject.kOverwrite)
    
    plot.success(f'Saved response matrix histograms to {rf_output_path}')
    return rf_output_path

def main(
    file_manager : utils.FileManager,
        sample : utils.Sample, 
        lepton_channel : int, 
        output : str = './output', 
        output_plots : str = None,
        vars : list[utils.Variable] = _default_vars,
        optimization_range : tuple[float, float] = None,
        variation = False,
    ):

    ### Output dir ###
    os.makedirs(output, exist_ok=True)
    output_plots = output_plots or output

    ### Files ###    
    rf_output_path = output_path(output, sample, lepton_channel)
    rf_output_file = ROOT.TFile(rf_output_path, 'RECREATE')

   

    
    create_response(
            file_manager=file_manager,
            sample=sample, 
            lepton_channel=lepton_channel, 
            rf_output_file=rf_output_file,
            output_plots = output_plots,
            rf_output_path=rf_output_path,
            vars=vars,
            optimization_range=optimization_range,
            variation=False,
            variation_name="",
            )
    if variation:
        
        variations_list = []

        for x in utils.variations_hist:
            variations_list.append(x+ utils.variation_up_key)
            variations_list.append(x+ utils.variation_down_key)
        for x in utils.Variation_hist_onesided:
            variations_list.append(x+ utils.variation_up_key)

        for variation_name in variations_list:
            create_response(
                file_manager=file_manager,
                sample=sample, 
                lepton_channel=lepton_channel, 
                rf_output_file=rf_output_file,
                output_plots = output_plots,
                rf_output_path=rf_output_path,
                vars=vars,
                optimization_range=optimization_range,
                variation=variation,
                variation_name=variation_name
            )
    


if __name__ == "__main__":
    ### Args ###
    parser = ArgumentParser(
        description="Plots the migration matrix, efficiency and fiducial accuracy. Saves the response matrix as histograms for use in ResonanceFinder.", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('filepath', help='Path to the CxAODReader output histograms')
    parser.add_argument('sample', help='Sample name as in [utils.Sample]')
    parser.add_argument("lepton", type=int, choices=[0, 1, 2])
    parser.add_argument('-o', '--output', default='./output')
    parser.add_argument('--optimize', help='Automatically optimize the binning. Pass in a "min,max" range of values that the binning should cover.')
    parser.add_argument('--variation', action='store_true', help='create the unfolding matrix for all variations')
    args = parser.parse_args()

    ### Files ###
    sample = utils.Sample.parse(args.sample)
    file_manager = utils.FileManager(
        samples=[sample],
        # file_path_formats=[args.filepath],
        lepton_channels=[args.lepton],
        file_path_formats = ['/eos/user/a/anubhav/phd/condor_merge/{lep}_diboson_x_new.root']
    )
    ### Run ###
    plot.file_formats = ['png', 'pdf']
    plot.save_transparent_png = False
    main(
        file_manager=file_manager,
        sample=sample, 
        lepton_channel=args.lepton,
        output=args.output,
        optimization_range=None if args.optimize is None else [float(x) for x in args.optimize.split(',')],
        variation = args.variation # to create the unfolding matrix for all variations
    )