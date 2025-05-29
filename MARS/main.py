from MARS.utils import normalize_dataframe, save_dataframes, combine_metrics
from MARS.operations import load_input_and_preprocess, check_df_absolute_or_relative_counts, remove_clades_from_taxaNames, concatenate_genus_and_species_names, split_taxonomic_groups, rename_taxa, calculate_metrics, check_presence_in_modelDatabase, filter_samples_low_read_counts
from logging_config import setup_logger
import pandas as pd
import os
import logging


def process_microbial_abundances(input_file1, input_file2, output_path=None, cutoff=0.000001, output_format="csv", stratification_file=None, flagLoneSpecies=False, taxaSplit=";", removeCladeExtensionsFromTaxa=True, whichModelDatabase="full_db", userDatabase_path="", sample_read_counts_cutoff=1):
    # Initialize logger to generate a MARS log file    
    logger = setup_logger('main', os.path.join(output_path, 'MARS.log'))
    logger_taxa_below_cutoff = setup_logger('taxa_below_cutoff', os.path.join(output_path, 'MARS_taxaBelowCutoff.log'))
    
    logger.info(f'INPUT VARIABLES - taxaSplit: {taxaSplit}, flagLoneSpecies: {flagLoneSpecies}, sample_read_counts_cutoff: {sample_read_counts_cutoff}, cutoff: {cutoff}, removeCladeExtensionsFromTaxa: {removeCladeExtensionsFromTaxa}, whichModelDatabase: {whichModelDatabase}, userDatabase_path (if whichModelDatabase is set to "user_db"): {userDatabase_path}.')
    
    # Run MARS
    # Step 1: Check input data & preprocess
    [preprocessed_dataframe, dfvalues_are_rel_abundances] = load_input_and_preprocess(input_file1, input_file2, taxaSplit=taxaSplit, input_is_df=False)

    # Optional Step: Remove potential clade extensions (e.g. "clade A"; " A") from taxa namings if set true
    if removeCladeExtensionsFromTaxa == True:
        preprocessed_dataframe = remove_clades_from_taxaNames(preprocessed_dataframe, taxaSplit=taxaSplit)
    
    # Optional Step: Concatenate genus name with species epithet if both are present, otherwise leave species column unchanged
    if flagLoneSpecies:
        preprocessed_dataframe = concatenate_genus_and_species_names(preprocessed_dataframe, taxaSplit=taxaSplit)

    # Step 2: Rename taxa according to resources/renaming.json to share same nomenclature as the model-database
    renamed_dataframe = rename_taxa(preprocessed_dataframe)

    # Step 3: Filter out samples with too few total reads from subsequent analysis in case input 
    # is in absolute read counts (not relative abundance)
    if dfvalues_are_rel_abundances == False:
        renamed_dataframe = filter_samples_low_read_counts(renamed_dataframe, sample_read_counts_cutoff=sample_read_counts_cutoff)
    
    # Step 4: Normalize & apply taxa relative abundance cutoff to renamed_dataframe
    logger.info('Normalizing & applying the species relative abundance cutoff to the pre-mapping dataframe.')
    [dataframe_afterCutoff, normalized_dataframe_afterCutoff] = normalize_dataframe(renamed_dataframe, dfvalues_are_rel_abundances=dfvalues_are_rel_abundances, cutoff=cutoff)

    # Step 5: Check for presence of input taxa in a specified model database (AGORA2, APOLLO, 
    # combination of both or user-defined)
    present_dataframe, absent_dataframe = check_presence_in_modelDatabase(dataframe_afterCutoff, whichModelDatabase=whichModelDatabase, userDatabase_path=userDatabase_path, taxaSplit=taxaSplit)
    if present_dataframe.empty:
        logger.critical('No species from the input data were found & could be mapped to the reconstruction database. Led to termination of code execution with ValueError.')
        raise ValueError("No species from the input data were found & could be mapped to the reconstruction database.")

    # Step 6.1: Normalize present_dataframe & absent_dataframe (with cutoff = 0,
    # because cutoff was already applied on dataframe_afterCutoff)
    logger.info('Normalizing the post-mapping dataframes (present & absent).')
    [present_dataframe_afterCutoff, normalized_present_dataframe] = normalize_dataframe(present_dataframe, dfvalues_are_rel_abundances=dfvalues_are_rel_abundances, cutoff=0, dataframe_to_normalize_to=dataframe_afterCutoff)
    [absent_dataframe_afterCutoff, normalized_absent_dataframe] = normalize_dataframe(absent_dataframe, dfvalues_are_rel_abundances=dfvalues_are_rel_abundances, cutoff=0, dataframe_to_normalize_to=dataframe_afterCutoff)
    # Step 6.2: Additionally normalize present_dataframe to its own total read count to be valid input for modelling 
    # (with relative abundances of present species per sample summing to 1)
    [_, normalized_present_dataframe_adjForModelling] = normalize_dataframe(present_dataframe, dfvalues_are_rel_abundances=dfvalues_are_rel_abundances, cutoff=0)

    # Step 7.1: Seperate normalized dataframes by taxonomic levels 
    # (one df per taxonomic level, for both normalized & not-normalized dataframes)
    preMapped_dataframes = split_taxonomic_groups(dataframe_afterCutoff, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    present_dataframes = split_taxonomic_groups(present_dataframe_afterCutoff, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    absent_dataframes = split_taxonomic_groups(absent_dataframe_afterCutoff, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    preMapped_dataframes_normalized = split_taxonomic_groups(normalized_dataframe_afterCutoff, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    present_dataframes_normalized = split_taxonomic_groups(normalized_present_dataframe, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    absent_dataframes_normalized = split_taxonomic_groups(normalized_absent_dataframe, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)
    present_dataframes_adjForModelling = split_taxonomic_groups(normalized_present_dataframe_adjForModelling, flagLoneSpecies=flagLoneSpecies, taxaSplit=taxaSplit)

    # Step 7.2: Add "pan" prefix to species names (in index) of present_dataframes_adjForModelling 
    # (because pan-species reconstructions will be used in MgPipe)
    present_dataframes_adjForModelling['Species'].index = 'pan' + present_dataframes_adjForModelling['Species'].index

    # Step 8.1: Calculate metrics on mapping coverage & microbiome composition
    logger.info('Calculating metrices for pre-mapping dataframes.')
    pre_mapping_metrics, pre_mapping_abundance_metrics, pre_mapping_beta_diversity, pre_mapping_summ_stats = calculate_metrics(preMapped_dataframes_normalized, preMapped_dataframes)
    logger.info('Calculating metrices for post-mapping present dataframes.')
    present_post_mapping_metrics, present_post_mapping_abundance_metrics, present_post_mapping_beta_diversity, present_post_mapping_summ_stats = calculate_metrics(present_dataframes_adjForModelling, present_dataframes)
    logger.info('Calculating metrices for post-mapping absent dataframes.')
    absent_post_mapping_metrics, absent_post_mapping_abundance_metrics, absent_post_mapping_beta_diversity, absent_post_mapping_summ_stats = calculate_metrics(absent_dataframes_normalized, absent_dataframes)
    
    # Step 8.2: Combine pre- and postMapping information of metrices, where needed
    logger.info('Combining metrics dataframes.')
    combined_metrics = combine_metrics(pre_mapping_metrics, present_post_mapping_metrics, df_type="metrics", dfvalues_are_rel_abundances=dfvalues_are_rel_abundances)
    combined_summ_stats = combine_metrics(pre_mapping_summ_stats, present_post_mapping_summ_stats, df_type="summ_stats", dfvalues_are_rel_abundances=dfvalues_are_rel_abundances)
    
    # Optional Step: If stratification groups are provided, stratify dataframe on these 
    # groups & calculate metrices on them too
    stratification_groups = {}
    stratification_groupnames = []
    if stratification_file is not None:
        stratification = pd.read_csv(stratification_file)
        for group in stratification["group"].unique():
            # Select the columns for this group
            group_columns = list(stratification[stratification["group"] == group]["samples"])
            pre_group_metrics, pre_group_abundance_metrics, pre_group_summ_stats = calculate_metrics(preMapped_dataframes_normalized, preMapped_dataframes, group=group_columns)
            post_group_metrics, post_group_abundance_metrics, post_group_summ_stats = calculate_metrics(present_dataframes_normalized, present_dataframes, group=group_columns)

            combined_group_metrics = combine_metrics(pre_group_metrics, post_group_metrics, df_type="metrics")
            combined_group_summ_stats = combine_metrics(pre_group_summ_stats, post_group_summ_stats, df_type="summ_stats")
            group_name = f"{group.lower()}_stratified_metrics"
            stratification_groupnames.append(group_name)
            stratification_groups[group_name] = [combined_group_metrics, combined_group_summ_stats, \
                                                pre_group_abundance_metrics, post_group_abundance_metrics]
        logger.info(f'Stratifying taxa dataframes using following groups: {stratification_groupnames}.')

    # Step 9: Store all result dataframes in a structure & save, if output-path is provided
    dataframe_groups = {'normalized_preMapped': preMapped_dataframes_normalized, 
                        'renormalized_mapped_forModelling': present_dataframes_adjForModelling,
                        'normalized_mapped': present_dataframes_normalized,
                        'normalized_unmapped': absent_dataframes_normalized,
                        'metrics': [combined_metrics, combined_summ_stats, \
                                    pre_mapping_abundance_metrics, \
                                   present_post_mapping_abundance_metrics, \
                                   absent_post_mapping_abundance_metrics, \
                                   pre_mapping_beta_diversity, \
                                   present_post_mapping_beta_diversity, \
                                   absent_post_mapping_beta_diversity]}
    
    dataframe_groups.update(stratification_groups)

    if output_path is not None:
        logger.info(f'Saving output to {output_path}.')
        save_dataframes(dataframe_groups, output_path, output_format)

        renamed_dataframe.to_csv(os.path.join(output_path, 'preprocessedInput_afterRenaming.csv'), sep=',')


    return dataframe_groups
