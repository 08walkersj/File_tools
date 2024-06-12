import os
import pandas as pd
from General_Tools.user_input_tools import validinput


def load_all(folder, suffix='.csv', **read_args):
    """
    Load all files in a folder into a single pandas DataFrame.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the files to be loaded.
    suffix : str, optional
        Suffix of the files to be loaded. Default is '.csv'.
    **read_args : dict
        Additional keyword arguments to pass to pandas.read_csv().

    Returns:
    --------
    pandas.DataFrame
        Concatenated DataFrame containing data from all loaded files.

    Notes:
    ------
    - The function reads all files in the specified folder with the given suffix
      and concatenates them into a single DataFrame.
    - Additional parameters for reading files can be specified using **read_args.

    """
    files= os.listdir(folder)
    files.sort()
    dfs= [pd.read_csv(folder+file, **read_args) for file in files if file.endswith(suffix)]
    return pd.concat(dfs)
def subset_to_pd_hdf(folder, files, out_path='./all.hdf5', suffix='.csv', read_args={'mode':'r'}, **save_kwargs):
    defaults= dict(key='main',mode='a',append=True,format='t', data_columns=True)
    defaults.update(save_kwargs)
    dfs= [pd.read_csv(folder+file, **read_args) for file in files if file.endswith(suffix)]
    pd.concat(dfs).to_hdf(out_path, **defaults)
def all_to_pd_hdf(folder, out_path='./all.hdf5', small=False, suffix='.csv', read_args={'mode':'r'}, **save_kwargs):
    """
    Convert CSV files in a folder to a single HDF5 file.

    Parameters:
    -----------
    folder : str
        Path to the folder containing the files to be converted.
    out_path : str, optional
        Path to the output HDF5 file. Default is './all.hdf5'.
    small : bool, optional
        If True, loads all CSV files into memory before saving to HDF5. If False,
        each CSV file is loaded and saved individually. Default is False.
    suffix : str, optional
        Suffix of the files to be converted. Default is 'csv'.
    read_args : dict, optional
        Additional keyword arguments to pass to pandas.read_csv(). Default is {'mode': 'r'}.
    **save_kwargs : dict
        Additional keyword arguments to pass to pandas.DataFrame.to_hdf().

    Raises:
    -------
    FileExistsError
        If the output file already exists and the user chooses not to continue appending data.

    Notes:
    ------
    - If 'small' is True, it's recommended to ensure that the combined size of all CSV files
      and available memory are manageable.
    - 'save_kwargs' can be used to specify parameters like 'key', 'mode', 'append', 'format',
      and 'data_columns' for saving to HDF5.

    """
    import os
    if os.path.isfile(out_path):
        if not validinput('Out file already exists continuing will append to the current file if possible. Continue? (y/n)', 'y', 'n'):
            raise FileExistsError('Please remove the current file or provide a new out file path if you do not wish to append')
    defaults= dict(key='main',mode='a',append=True,format='t', data_columns=True)
    defaults.update(save_kwargs)
    if small:
        load_all(folder, suffix=suffix, **read_args).to_hdf(out_path, **defaults)
    else:
        import gc
        files= os.listdir(folder)
        files.sort()
        for file in files: 
            if file.endswith(suffix):
                pd.read_csv(folder+file, **read_args).to_hdf(out_path, **defaults)
                gc.collect()
def to_pd_hdf(folder, out_path='./all.hdf5', chunks=0, suffix='.csv', read_args={'mode':'r'}, **save_kwargs):
    import os
    if os.path.isfile(out_path):
        if not validinput('Out file already exists continuing will append to the current file if possible. Continue? (y/n)', 'y', 'n'):
            raise FileExistsError('Please remove the current file or provide a new out file path if you do not wish to append')

    if not chunks:
        all_to_pd_hdf(folder, out_path='./all.hdf5', small=True, suffix=suffix, read_args={'mode':'r'}, **save_kwargs)
    else:
        files= os.listdir(folder)
        files.sort()
        steps= int(len(files)/chunks)
        for i in range(chunks-1):
            subset= files[i*steps:(i+1)*steps]
            subset_to_pd_hdf(folder, subset, out_path=out_path, suffix=suffix, read_args=read_args, **save_kwargs)
        subset= files[i*steps:]
        subset_to_pd_hdf(folder, subset, out_path=out_path, suffix=suffix, read_args=read_args, **save_kwargs)
