import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger('main.utils')
logger_taxa_below_cutoff = logging.getLogger('taxa_below_cutoff.utils')

def read_file_as_dataframe(file_path, header=0, index_col=None):
    """
    Reads a file into a pandas DataFrame based on its extension.

    Args:
        file_path (str): Path to the file.
        header (int): Row number to use as column names (default is 0).
        index_col (int or None): Column to use as the row labels of the DataFrame (default is None).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the file.
    """
    # Extract the file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    # Map extensions to file types
    try:
        file_extension = file_path.type
    except AttributeError:
        ind = file_path.find('.')
        extension = file_path[ind+1:]
        if extension == 'txt':
            file_extension = 'text/plain'
        elif extension == 'csv':
            file_extension = 'text/csv'
        elif extension == 'xlsx':
            file_extension = 'spreadsheet'
        else:
            file_extension = 'not found'

    # Read the file based on its type
    if file_extension in ['text/plain', 'text/csv', 'text/tab-separated-values']:
        # Assuming text files are tab-separated by default; adjust delimiter as needed
        delimiter = '\t' if file_extension == 'text/plain' else ','
        return pd.read_csv(file_path, sep=delimiter, index_col=index_col, low_memory=False, header=header)
    elif file_extension == 'spreadsheet'or file_extension == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return pd.read_excel(file_path, engine='openpyxl', index_col=index_col, header=header)
    elif file_extension == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def merge_files(input1, input2, input_is_df=False):
    """
    Merge two files using pandas and return the merged DataFrame.

    Args:
        input1 (str or pd.DataFrame): The first input, either a file path or a DataFrame.
        input2 (str or pd.DataFrame): The second input, either a file path or a DataFrame, or None.
        input_is_df (bool): Whether the inputs are DataFrames instead of file paths.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """

    # Read input files into pandas DataFrames  
    if input2 is None:
        if not input_is_df:
            df1 = read_file_as_dataframe(input1, header=0, index_col=0)
        else:
            df1 = input1

        merged_df = df1

        if merged_df.index.name != 'Taxon':
            if 'Taxon' in merged_df.columns:
                merged_df = merged_df.set_index('Taxon')
    else:
        if not input_is_df:
            df1 = read_file_as_dataframe(input1, header=0, index_col=0)
            df2 = read_file_as_dataframe(input2, header=1, index_col=0)
        else:
            df1 = input1
            df2 = input2

        # Merge DataFrames using their index values
        merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

        try:
            merged_df = merged_df.drop(columns=['Confidence'])
        except KeyError:
            pass

        merged_df = merged_df.reset_index(drop=True)

        if 'Taxon' in merged_df.columns:
            merged_df = merged_df.set_index('Taxon')

    return merged_df


def normalize_dataframe(dataframe, dfvalues_are_rel_abundances=False, cutoff=0.000001, dataframe_to_normalize_to=None):
    """
    Normalize a dataframe by first grouping and summing rows with the same name,
    and then calculating the relative abundances per taxa so that the sum of each sample (each column) is 1.
    Optionally, a cutoff can be provided to filter out low abundant taxa after normalization, as well as 
    an additional dataframe containing total read counts for all samples of the input dataframe, to which, in case
    being provided, will be normalized to instead.

    Args:
        dataframe (pd.Dataframe):                           The input DataFrame with taxonomic groups in the index and read counts per sample.
        dfvalues_are_rel_abundances (boolean, optional):    Flag indicating whether read counts are already normalized and therefore
                                                            only the cutoff will be applied.
                                                            Defaults to False.
        cutoff (float, optional):                           A cut-off value for filtering out low abundance taxa. Defaults to None.
        dataframe_to_normalize_to (pd.Dataframe, optional): Containing per-sample total read counts pre-mapping,
                                                            allowing for taxa abundance normalization against pre-mapped total read counts
                                                            instead of normalization against only present or absent total read counts.
                                                            Defaults to None.

    Returns:
        grouped_df_afterCutoff (pd.Dataframe): The input DataFrame with taxonomic groups in the index and total read counts per sample, after cutoff was applied.
        rel_abundances_df_afterCutoff (pd.Dataframe): The input DataFrame with taxonomic groups in the index and normalized relative abundances per sample, after cutoff was applied.
    """
    # Group by index and sum the rows with the same name
    grouped_df = dataframe.groupby(dataframe.index.name).sum()
    
    # Normalize each column so that the sum of each column is 1 (either
    # to pre-mapped total read counts, or to the subset read counts for 
    # the dataset with taxa present in model database - which needs to be done for modelling to work)
    if dataframe_to_normalize_to is not None:
        read_counts = dataframe_to_normalize_to.sum()
    else:
        read_counts = grouped_df.sum()
    
    # Normalize read counts to get relative abundances of taxa
    rel_abundances_df = grouped_df.div(read_counts)

    # Apply cut-off for low abundance taxa
    if cutoff != 0:
        # Identify which taxa are below relative abundance cutoff threshold & set their read counts to 0
        grouped_df[rel_abundances_df <= cutoff] = 0

        # Remove taxa which are non-abundant in any sample after cutoff has been applied
        grouped_df_afterCutoff = grouped_df[(grouped_df != 0).any(axis=1)]

        # Renormalize the dataframe after cutoff was applied so that relative abundances per sample sum to 1
        read_counts_afterCutoff = grouped_df_afterCutoff.sum()
        rel_abundances_df_afterCutoff = grouped_df_afterCutoff.div(read_counts_afterCutoff)

        # Log which taxa are below the cutoff in which samples and were set to 0
        entries_below_cutoff = rel_abundances_df[rel_abundances_df <= cutoff].stack().index.tolist()
        if entries_below_cutoff:
            logger.info(f"Taxa were below the cutoff & are listed in seperate log-file.")
            logger_taxa_below_cutoff.info(f"Taxa whose rel.abundance was below the cutoff & therefore set to 0: {entries_below_cutoff}")
        else:
            logger.info(f"No taxa were below the cutoff.")
    else:
        grouped_df_afterCutoff = grouped_df
        rel_abundances_df_afterCutoff = rel_abundances_df
        
    return grouped_df_afterCutoff, rel_abundances_df_afterCutoff


def combine_metrics(metrics1, metrics2, df_type="metrics", dfvalues_are_rel_abundances=False):
    """
    Combine the metrics from two different sets of taxonomic DataFrames into a single DataFrame for each level.

    Args:
        metrics1 (dict): A dictionary with keys as taxonomic levels and values as the calculated metrics for the first group.
        metrics2 (dict): A dictionary with keys as taxonomic levels and values as the calculated metrics for the second group.

    Returns:
        dict: A dictionary with keys as taxonomic levels and values as the combined DataFrames.
    """
    combined_metrics = {}

    for level in metrics1.keys():
        level_metrics_pre_mapping = metrics1[level]
        level_metrics_post_mapping = metrics2[level]

        if df_type == "metrics":
            combined_level_metrics = {}

            for metric_name in level_metrics_pre_mapping.keys():
                
                combined_metric = pd.DataFrame([level_metrics_pre_mapping[metric_name], level_metrics_post_mapping[metric_name]])
                combined_metric.index = ['Pre mapping', 'Post mapping']
                combined_level_metrics[metric_name] = combined_metric

            combined_metrics[level] = combined_level_metrics

        elif df_type == "summ_stats":
            combined_metric = pd.DataFrame({'Pre mapping': level_metrics_pre_mapping, \
                                            'Post mapping': level_metrics_post_mapping})
            
            combined_metric['Mapping coverage'] = np.nan
            combined_metric.loc[0, 'Mapping coverage'] = combined_metric.loc[0, 'Post mapping'] / combined_metric.loc[0, 'Pre mapping']
            combined_metric.loc[2, 'Mapping coverage'] = combined_metric.loc[2, 'Post mapping'] / combined_metric.loc[2, 'Pre mapping']
            combined_metric.loc[4, 'Mapping coverage'] = combined_metric.loc[4, 'Post mapping'] - combined_metric.loc[4, 'Pre mapping']
            combined_metric.loc[6, 'Mapping coverage'] = combined_metric.loc[6, 'Post mapping'] / combined_metric.loc[6, 'Pre mapping']
            
            if dfvalues_are_rel_abundances == False:
                combined_metric['Description'] = ['The number of taxa across all samples. MappingCoverage = post mapping/pre mapping.', \
                                                  'The estimated number of taxa following standard nomenclature. Estimated by excluding all taxa whose names contain "-" &/or multiple uppercase letters in a row.', \
                                                  'Mean number of taxa across samples (equals to species richness on species taxonomic level). MappingCoverage = post mapping/pre mapping.', \
                                                  'Standard deviation of taxa richness.', 'Mean alpha-diversity (calc. by pielous evenness) in samples. Towards 0: low diversity, towards 1: high diversity, with 1: complete evenness. MappingCoverage = post mapping - pre mapping.', \
                                                 'Standard deviation of pielous evenness.', 'Mean number of reads across samples. MappingCoverage = post mapping/pre mapping.', \
                                                 'Standard deviation of read counts.']
    
                combined_metric.index = ['Total number of taxa', 'Estimated total number of named taxa', 'Mean taxa richness', \
                                         'Std taxa richness', 'Mean pielous evenness', 'Std pielous evenness', \
                                         'Mean read counts', 'Std read counts']
            else:
                combined_metric['Description'] = ['The number of taxa across all samples. MappingCoverage = post mapping/pre mapping.', \
                                                  'The estimated number of taxa following standard nomenclature. Estimated by excluding all taxa whose names contain "-" &/or multiple uppercase letters in a row.', \
                                                  'Mean number of taxa across samples (equals to species richness on species taxonomic level). MappingCoverage = post mapping/pre mapping.', \
                                                  'Standard deviation of taxa richness.', 'Mean alpha-diversity (calc. by pielous evenness) in samples. Towards 0: low diversity, towards 1: high diversity, with 1: complete evenness. MappingCoverage = post mapping - pre mapping.', \
                                                 'Standard deviation of pielous evenness.', 'Mean relative abundance across samples. MappingCoverage = post mapping/pre mapping.', \
                                                 'Standard deviation of relative abundance.']

                combined_metric.index = ['Total number of taxa', 'Estimated total number of named taxa', 'Mean taxa richness', \
                                         'Std taxa richness', 'Mean shannon index', 'Std shannon index', \
                                         'Mean relative abundance', 'Std relative abundance']
            
            combined_metrics[level] = combined_metric

    return combined_metrics


def save_dataframes(dataframe_groups, output_path, output_format):
    """
    Save DataFrames to the specified output path in the given format.

    Args:
        dataframe_groups (dict): A dictionary of dictionaries containing DataFrames.
                                  Outer dictionary keys represent group names, and inner dictionaries contain
                                  DataFrames for each taxonomic level.
        output_path (str): The directory path where the output files will be saved.
        output_format (str): The format to save the DataFrames ('csv', 'excel', 'parquet', or 'json').
    """
    os.makedirs(output_path, exist_ok=True)

    for group_name, values in dataframe_groups.items():
        group_output_path = os.path.join(output_path, group_name)
        os.makedirs(group_output_path, exist_ok=True)
        if group_name == "metrics":
            for level, metrics_dataframes in values[0].items():
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                for metric_name, df in metrics_dataframes.items():
                    file_name = f"{metric_name}.{output_format}"
                    file_path = os.path.join(level_output_path, file_name)

                    if output_format == "csv":
                        df.to_csv(file_path)
                    elif output_format == "txt" or output_format == "tsv":
                        df.to_csv(file_path, sep='\t')
                    elif output_format == "excel":
                        df.to_excel(file_path)
                    elif output_format == "parquet":
                        df.to_parquet(file_path)
                    elif output_format == "json":
                        df.to_json(file_path)
                    else:
                        raise ValueError(f"Unsupported output format: {output_format}")

            for level, summ_stats_df in values[1].items():
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"summary_statistics_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    summ_stats_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    summ_stats_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    summ_stats_df.to_excel(file_path)
                elif output_format == "parquet":
                    summ_stats_df.to_parquet(file_path)
                elif output_format == "json":
                    summ_stats_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            for level, abundance_metrics_df in values[2].items():
                # Replace ' ' by '_' in taxa names
                abundance_metrics_df.index = abundance_metrics_df.index.str.replace(' ', '_')

                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"preMapping_abundanceMetrics_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    abundance_metrics_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    abundance_metrics_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    abundance_metrics_df.to_excel(file_path)
                elif output_format == "parquet":
                    abundance_metrics_df.to_parquet(file_path)
                elif output_format == "json":
                    abundance_metrics_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            for level, abundance_metrics_df in values[3].items():
                # Replace ' ' by '_' in taxa names
                abundance_metrics_df.index = abundance_metrics_df.index.str.replace(' ', '_')

                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"mapped_abundanceMetrics_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    abundance_metrics_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    abundance_metrics_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    abundance_metrics_df.to_excel(file_path)
                elif output_format == "parquet":
                    abundance_metrics_df.to_parquet(file_path)
                elif output_format == "json":
                    abundance_metrics_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
    
            for level, abundance_metrics_df in values[4].items():
                # Replace ' ' by '_' in taxa names
                abundance_metrics_df.index = abundance_metrics_df.index.str.replace(' ', '_')

                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"unmapped_abundanceMetrics_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    abundance_metrics_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    abundance_metrics_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    abundance_metrics_df.to_excel(file_path)
                elif output_format == "parquet":
                    abundance_metrics_df.to_parquet(file_path)
                elif output_format == "json":
                    abundance_metrics_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            for level, beta_diversity_df in values[5].items():
                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"preMapping_brayCurtisDissimilarity_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    beta_diversity_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    beta_diversity_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    beta_diversity_df.to_excel(file_path)
                elif output_format == "parquet":
                    beta_diversity_df.to_parquet(file_path)
                elif output_format == "json":
                    beta_diversity_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
        
            for level, beta_diversity_df in values[6].items():
                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"mapped_brayCurtisDissimilarity_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    beta_diversity_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    beta_diversity_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    beta_diversity_df.to_excel(file_path)
                elif output_format == "parquet":
                    beta_diversity_df.to_parquet(file_path)
                elif output_format == "json":
                    beta_diversity_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
        
            for level, beta_diversity_df in values[7].items():
                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"unmapped_brayCurtisDissimilarity_{level}.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    beta_diversity_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    beta_diversity_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    beta_diversity_df.to_excel(file_path)
                elif output_format == "parquet":
                    beta_diversity_df.to_parquet(file_path)
                elif output_format == "json":
                    beta_diversity_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

        elif "_stratified_metrics" in group_name:
            for level, metrics_dataframes in values[0].items():
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                for metric_name, df in metrics_dataframes.items():
                    file_name = f"{metric_name}_stratified.{output_format}"
                    file_path = os.path.join(level_output_path, file_name)

                    if output_format == "csv":
                        df.to_csv(file_path)
                    elif output_format == "txt" or output_format == "tsv":
                        df.to_csv(file_path, sep='\t')
                    elif output_format == "excel":
                        df.to_excel(file_path)
                    elif output_format == "parquet":
                        df.to_parquet(file_path)
                    elif output_format == "json":
                        df.to_json(file_path)
                    else:
                        raise ValueError(f"Unsupported output format: {output_format}")

            for level, summ_stats_df in values[1].items():
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"summary_statistics_{level}_stratified.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    summ_stats_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    summ_stats_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    summ_stats_df.to_excel(file_path)
                elif output_format == "parquet":
                    summ_stats_df.to_parquet(file_path)
                elif output_format == "json":
                    summ_stats_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            for level, abundance_metrics_df in values[2].items():
                # Replace ' ' by '_' in taxa names
                abundance_metrics_df.index = abundance_metrics_df.index.str.replace(' ', '_')

                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"preMapping_abundanceMetrics_{level}_stratified.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    abundance_metrics_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    abundance_metrics_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    abundance_metrics_df.to_excel(file_path)
                elif output_format == "parquet":
                    abundance_metrics_df.to_parquet(file_path)
                elif output_format == "json":
                    abundance_metrics_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            for level, abundance_metrics_df in values[3].items():
                # Replace ' ' by '_' in taxa names
                abundance_metrics_df.index = abundance_metrics_df.index.str.replace(' ', '_')

                # Save dataframes
                level_output_path = os.path.join(group_output_path, level)
                os.makedirs(level_output_path, exist_ok=True)

                file_name = f"mapped_abundanceMetrics_{level}_stratified.{output_format}"
                file_path = os.path.join(level_output_path, file_name)

                if output_format == "csv":
                    abundance_metrics_df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    abundance_metrics_df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    abundance_metrics_df.to_excel(file_path)
                elif output_format == "parquet":
                    abundance_metrics_df.to_parquet(file_path)
                elif output_format == "json":
                    abundance_metrics_df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
    
        else:
            for level, df in values.items():
                # Replace ' ' by '_' in taxa names
                df.index = df.index.str.replace(' ', '_')

                # Save dataframes
                file_name = f"{group_name}_{level.lower()}.{output_format}"
                file_path = os.path.join(group_output_path, file_name)

                if output_format == "csv":
                    df.to_csv(file_path)
                elif output_format == "txt" or output_format == "tsv":
                    df.to_csv(file_path, sep='\t')
                elif output_format == "excel":
                    df.to_excel(file_path)
                elif output_format == "parquet":
                    df.to_parquet(file_path)
                elif output_format == "json":
                    df.to_json(file_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
            
