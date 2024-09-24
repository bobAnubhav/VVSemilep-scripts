#!/usr/bin/env python3
'''
@file rf_plu.py 
@author Riley Xu - riley.xu@gmail.com, riley.xu@cern.ch 
@date February 24, 2024 
@brief Script for running ResonanceFinder fits.

Note this handles the diboson x-sec and EFT limit fits too, not just PLU.
'''

from typing import Union

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil

import utils
from plotting import plot
import eft_quad_correction

##########################################################################################
###                                       CONFIG                                       ###
##########################################################################################

def ws_path(output_dir, base_name, mode, stat_validation_index=None):
    if stat_validation_index is None:
        return f'{output_dir}/rf/{base_name}.{mode}_ws.root'
    else:
        return f'{output_dir}/rf/{base_name}.{mode}_ws_var{stat_validation_index:03}.root'



##########################################################################################
###                                        MODES                                       ###
##########################################################################################

def _add_plu(runner, lepton_channel, variable, region, response_matrix_path, **_):
    bins = utils.get_bins(lepton_channel, variable)
    nbins = len(bins) - 1
    signal_names = ''
    for i in range(1, nbins + 1):
        i_str = str(i).rjust(2, '0')
        sigName = "bin" + i_str
        poiName = "mu_" + i_str
        if i > 1:
            signal_names += '-'
        signal_names += sigName

        runner.channel(region).addSample(sigName, response_matrix_path.format(lep=lepton_channel), f'ResponseMatrix_{variable}_fid{i_str}')
        runner.channel(region).sample(sigName).multiplyBy(poiName, 100, 0, 1e6)
        for variation in (utils.variations_hist if lepton_channel != 0 else []):
            runner.channel(region).sample(sigName).addVariation(variation)
        for variation in (utils.Variation_hist_onesided if lepton_channel != 0 else []):
            runner.channel(region).sample(sigName).addVariation(variation)
            runner.addOneSideVariation(variation)
        # adding GPR correction for signal contamination
        
        runner.defineSignal(runner.channel(region).sample(sigName), 'Unfold')
        runner.addPOI(poiName)
    return signal_names


def _add_diboson(runner, lepton_channel, lumi_uncert, variable, region, hist_file_format, hist_name, gpr_mu_corrs, output_dir, **_):
    from ROOT import RF # type: ignore

    runner.channel(region).addSample('diboson', hist_file_format.format(lep=lepton_channel, sample='diboson'), hist_name.format('diboson'))
    sample = runner.channel(region).sample('diboson')

    mu_factor = RF.MultiplicativeFactor('mu-diboson', 1, 0, 5, RF.MultiplicativeFactor.FREE)
    sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN) # uncommented 
    sample.multiplyBy(mu_factor)
    sample.setUseStatError(True)
    for variation in (utils.variations_hist if lepton_channel != 0 else []): # TODO
        sample.addVariation(variation)
    for variation in (utils.Variation_hist_onesided if lepton_channel != 0 else []): # TODO 
        sample.addVariation(variation)
        runner.addOneSideVariation(variation)


    ### Add sig. contam. correction for GPR ###   
    if gpr_mu_corrs: 
        runner.channel(region).addSample('diff_pos', f'{output_dir}/gpr/gpr_{lepton_channel}lep_vjets_yield.root', f'gpr_mu-diboson_posdiff_{variable}')
        runner.channel(region).addSample('diff_neg', f'{output_dir}/gpr/gpr_{lepton_channel}lep_vjets_yield.root', f'gpr_mu-diboson_negdiff_{variable}')
        runner.channel(region).sample('diff_neg').multiplyBy(mu_factor)
        # Note the name "diff_neg" is really important so that ResonanceFinder/Root/BadBinFixer.cxx doesn't remove the histogram entirely

    runner.defineSignal(sample, 'diboson')
    runner.addPOI('mu-diboson')
    return 'diboson'


def _add_eft(runner, lepton_channel, lumi_uncert, region, hist_file_format, hist_name, mode, **_):
    '''
    @param mode
        Should be [cw]_[lin/quad].
    '''
    from ROOT import RF # type: ignore

    ### Add diboson as background ###
    runner.channel(region).addSample('diboson', hist_file_format.format(lep=lepton_channel, sample='diboson'), hist_name.format('diboson'))
    sample = runner.channel(region).sample('diboson')
    sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    for variation in (utils.variations_hist if lepton_channel != 0 else []): # TODO
        sample.addVariation(variation)
    for variation in (utils.Variation_hist_onesided if lepton_channel != 0 else []): # TODO
        sample.addVariation(variation)
        runner.addOneSideVariation(variation)


    ### Config ###
    signal_names = ''
    operator = mode.split('_')[0]

    ### Add linear term ###
    runner.channel(region).addSample(f'{operator}_lin', hist_file_format.format(lep=lepton_channel, sample=f'{operator}_lin'), hist_name.format(f'{operator}_lin'))
    sample = runner.channel(region).sample(f'{operator}_lin')
    sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
    sample.setUseStatError(True)
    # For some reason adding these variations don't work. Gives a really strong constraint
    # (like 0.003) and completely disagrees with no-systs, same in both lin and lin+quad.
    # Wonder if this has to do with the BadBinFixer? Or something similar.
    # for variation in utils.variations_hist:
    #     sample.addVariation(variation)
    
    lin_name = f'mu-{operator}-lin' if 'quad' in mode else f'mu-{operator}' # In quad mode, the postscript will create the true mu-cw variable. Names probably shouldn't collide.
    mu_factor = RF.MultiplicativeFactor(lin_name, 1, -5, 5, RF.MultiplicativeFactor.FREE)
    sample.multiplyBy(mu_factor)
    runner.defineSignal(sample, 'eft')
    runner.addPOI(lin_name)
    signal_names += f'{operator}_lin'

    ### Add quadratic term ###
    if 'quad' in mode:
        runner.channel(region).addSample(f'{operator}_quad', hist_file_format.format(lep=lepton_channel, sample=f'{operator}_quad'), hist_name.format(f'{operator}_quad'))
        sample = runner.channel(region).sample(f'{operator}_quad')
        sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
        sample.setUseStatError(True)
        for variation in (utils.variations_hist if lepton_channel != 0 else []): # TODO
            sample.addVariation(variation)
        for variation in (utils.Variation_hist_onesided if lepton_channel != 0 else []): # TODO
            sample.addVariation(variation)
            runner.addOneSideVariation(variation)

        
        mu_factor = RF.MultiplicativeFactor(f'mu-{operator}-quad', 1) # this is fixed by Rob's postscript in eft_quad_correction
        sample.multiplyBy(mu_factor)
        runner.defineSignal(sample, 'eft')
        signal_names += f'-{operator}_quad'

    return signal_names



##########################################################################################
###                                     COMMON RUN                                     ###
##########################################################################################
    
def run(
        mode : str,
        variables : dict[int, list[utils.Variable]],
        response_matrix_path : str,
        output_dir : str,
        base_name : str,
        hist_file_format : str,
        mu_stop : tuple[float, float],
        mu_ttbar : tuple[float, float],
        gpr_mu_corrs : bool = True,
        stat_validation_index : int = None,
    ):
    '''
    @param mode
        What should be set as the signal.
            - 'PLU': Runs the profile-likelihood unfolding fit, such that there are nbins
              signals, and the strength of each signal is the fiducial event count. These
              use the response matrix histograms as the signal inputs.
            - 'diboson': Runs a direct fit to the diboson signal strength, with the
              diboson MC as the only floating signal. Note this only looks at a single
              channel at a time though.
            - an EFT term: cw_lin/quad. Runs a fit to a single EFT operator, with the SM
              diboson signal strength fixed to 1 (TODO?)
    @param stat_validation_index
        Runs a validation test by using an alternate data sample, i.e. one that have been Poisson
        varied. The histograms should be named with prefix 'data_var{stat_validation_index:03}'.
    '''
    from ROOT import RF # type: ignore

    ### Config ###
    analysis = f'VV'
    outputWSTag = 'feb24'
    lumi_uncert = 0.017
    if stat_validation_index is not None:
        outputWSTag += f'_var{stat_validation_index:03}'
  
    ### Define the workspace ###
    runner = RF.DefaultAnalysisRunner(analysis)
    runner.setOutputDir(output_dir + '/rf')
    runner.setOutputWSTag(outputWSTag)
    runner.setCollectionTagNominal('')
    runner.setCollectionTagUp(utils.variation_up_key)
    runner.setCollectionTagDown(utils.variation_down_key)
    # VVUnfold.setPruningThreshold(0.01, 0.01) # is this buggy? According to Liza on Mattermost

    ### Regions ###
    for lep,var_list in variables.items():
        for variable in var_list:
            ### Setup ###
            region = f"Region_{lep}lep_MergedHP_SR_{variable}"
            runner.addChannel(region)
            runner.channel(region).setStatErrorThreshold(0.05) # 0.05 means that errors < 5% will be ignored

            ### Get hists ###
            hist_name = f'{{}}_VV{lep}Lep_MergHP_Inclusive_SR_' + utils.generic_var_to_lep(variable, lep).name

            ### Add data ###
            if stat_validation_index == None:
                hist_name_data = hist_name.format('data')
            else:
                hist_name_data = hist_name.format(f'data_var{stat_validation_index:03}')
            runner.channel(region).addData('data', hist_file_format.format(lep=lep, sample='data'), hist_name_data)

            ### Add ttbar ###
            runner.channel(region).addSample('ttbar', hist_file_format.format(lep=lep, sample='ttbar'), hist_name.format('ttbar'))
            sample = runner.channel(region).sample('ttbar')
            # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
            # fix by scaling by the mean as a constant first (which works fine).
            sample.multiplyBy('mu-ttbar_nom', mu_ttbar[0])
            sample.multiplyBy(utils.variation_mu_ttbar, 1, 1 - mu_ttbar[1] / mu_ttbar[0], 1 + mu_ttbar[1] / mu_ttbar[0], RF.MultiplicativeFactor.GAUSSIAN)
            sample.multiplyBy(utils.variation_lumi, 1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
            sample.setUseStatError(True)
            for variation in (utils.variations_hist if lep != 0 else []): # TODO
                sample.addVariation(variation)
            for variation in (utils.Variation_hist_onesided if lep != 0 else []): # TODO
                sample.addVariation(variation)
                runner.addOneSideVariation(variation)
   

            ### Add stop ###
            runner.channel(region).addSample('stop', hist_file_format.format(lep=lep, sample='stop'), hist_name.format('stop'))
            sample = runner.channel(region).sample('stop')
            # ResonanceFinder has a major bug when using mean != 1 GAUSSIAN constraints. So hack
            # fix by scaling by the mean as a constant first (which works fine).
            sample.multiplyBy('mu-stop_nom', mu_stop[0])
            sample.multiplyBy(utils.variation_mu_stop, 1, 1 - mu_stop[1] / mu_stop[0], 1 + mu_stop[1] / mu_stop[0], RF.MultiplicativeFactor.GAUSSIAN)
            sample.multiplyBy(utils.variation_lumi,  1, 1 - lumi_uncert, 1 + lumi_uncert, RF.MultiplicativeFactor.GAUSSIAN)
            sample.setUseStatError(True)
            for variation in (utils.variations_hist if lep != 0 else []): # TODO
                sample.addVariation(variation)
            for variation in (utils.Variation_hist_onesided if lep != 0 else []): # TODO
                sample.addVariation(variation)
                runner.addOneSideVariation(variation)



            ### Add GPR ###
            runner.channel(region).addSample('vjets', f'{output_dir}/gpr/gpr_{lep}lep_vjets_yield.root', 'Vjets_SR_' + variable.name)
            sample = runner.channel(region).sample('vjets')
            sample.setUseStatError(True)
            if gpr_mu_corrs:
                for variation in utils.variations_custom:
                    sample.addVariation(variation)
                for variation in (utils.variations_hist if lep != 0 else []): # TODO
                    sample.addVariation(variation)
                for variation in (utils.Variation_hist_onesided if lep != 0 else []): # TODO
                    sample.addVariation(variation)
                    runner.addOneSideVariation(variation)
                    


            ### Mode switch (signals and diboson background) ###
            common_args = {
                'mode': mode,
                'runner': runner,
                'lepton_channel': lep,
                'variable': variable,
                'region': region,
                'hist_file_format': hist_file_format,
                'hist_name': hist_name,
                'lumi_uncert': lumi_uncert,
                'gpr_mu_corrs': gpr_mu_corrs,
                'output_dir': output_dir,
            }
            if mode == 'PLU':
                signal_name = _add_plu(**common_args, response_matrix_path=response_matrix_path)
            elif mode == 'diboson':
                signal_name = _add_diboson(**common_args)
            elif 'lin' in mode or 'quad' in mode:
                signal_name = _add_eft(**common_args)
            else:
                raise NotImplementedError(f'rf.py() unknown mode {mode}')

            runner.linearizeRegion(region) # What is this for?

    ### Global options ###
    #VVUnfold.reuseHist(True) # What is this for?
    runner.debugPlots(True)

    ### Make workspace ###
    if stat_validation_index is None:
        log_name = f'{output_dir}/rf/log.{base_name}.{mode}_rf.txt'
    else:
        log_name = f'{output_dir}/rf/log.{base_name}.{mode}_rf_var{stat_validation_index:03}.txt'
    with plot.redirect(log_name):
        runner.produceWS()

    ### Copy back ###
    rf_output_path = f'{output_dir}/rf/ws/{analysis}_{signal_name}_{outputWSTag}.root'
    target_path = ws_path(output_dir, base_name, mode, stat_validation_index)
    if 'quad' in mode:
        operator = mode.split('_')[0]
        eft_quad_correction.joinPOI(rf_output_path, target_path, operator)
    else:
        shutil.copyfile(rf_output_path, target_path)
    plot.success(f'Created workspace at {target_path}')
    return target_path


##########################################################################################
###                                        MAIN                                        ###
##########################################################################################

def parse_args():
    parser = ArgumentParser(
        description="", 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", required=True, choices=['PLU', 'diboson'])
    parser.add_argument("--lepton", required=True, help='Comman-separated list of lepton channels')
    parser.add_argument("--var", required=True, help='Variable to fit against; this must be present in the utils.py module. Can be a comma-separated list matching --lepton.')
    parser.add_argument('--mu-ttbar', default='1,0.01', help="The ttbar signal strength. Should be a comma-separated pair val,err. (Use 0.72,0.03 for data fits)")
    parser.add_argument('-o', '--output', default='./output', help='Output directory. The script will look for the histograms from the rebin, response_matrix, and gpr subdirectories.')
    return parser.parse_args()


def main():
    '''
    See file header.
    '''
    args = parse_args()
    variables = [getattr(utils.Variable, x) for x in args.var.split(',')]
    lepton_channels = [int(x) for x in args.lepton.split(',')]
    mu_ttbar = [float(x) for x in args.mu_ttbar.split(',')]
    run(
        mode=args.mode,
        lepton_channels=lepton_channels,
        variables=variables,
        response_matrix_path=f'{args.output}/response_matrix/diboson_{lep}lep_rf_histograms.root',
        output_dir=args.output,
        hist_file_format=f'{args.output}/rebin/{{lep}}lep_{{sample}}_rebin.root',
        mu_stop=(1, 0.05),
        mu_ttbar=mu_ttbar,
        # mu_ttbar=(0.72, 0.03),
    )

if __name__ == '__main__':
    main()