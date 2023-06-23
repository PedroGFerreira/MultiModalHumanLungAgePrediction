import pandas as pd

def load_and_process_metadata(metadata_tab_file, index_col="SAMPIDDOT"):
    metadata = pd.read_csv(metadata_tab_file,
                           sep="\t", header=0, index_col=index_col)
    metadata = metadata.drop(metadata.columns[0], axis=1)  # remove unnecessary column in our metadata
    # drop rows with any NaNs
    metadata = metadata[~(metadata.isna().values.any(axis=1))]
    # create new column with subject id
    metadata["subject_id"] = [".".join(idx.split(".")[0:2]).replace(".", "-") for idx in metadata.index]
    # add age bin information with bins of size 5
    metadata["age_bins"] = pd.cut(metadata["AGE"], labels=False,
                                 bins=list(range(metadata["AGE"].min()-1, metadata["AGE"].max()+5+1, 5)))
    age = metadata["AGE"]
    age_bins = metadata["age_bins"]
    age_bins.index = age_bins.index.str.replace('.', '-', regex=False)
    age.index = age.index.str.replace(".", "-", regex=False)
    metadata.index = metadata.index.str.replace(".", "-", regex=False)
    age = age.loc[age.index.isin(metadata.index)]
    metadata = metadata.loc[metadata.index.isin(age.index)]
    return metadata, age, age_bins
