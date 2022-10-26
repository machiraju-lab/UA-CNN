import pandas as pd
import re


def get_data(path_to_file):
    data = pd.read_csv(path_to_file, names=['path'])
    return data

def clean_data(data_df):
    data_df[['root', 'wsi', 'aug','tile', 'to_del']] = data_df['path'].str.split('/', expand=True)
    data_df = data_df.drop(columns=["root", "to_del"])
    data_df["label"] = data_df["tile"].apply(lambda row: re.search('tn_(.*?)_rm', row).group(1) if re.search('tn_(.*?)_rm', row) else "")
    print(data_df["label"].unique())
    return data_df

def get_metadata():
    path = "all_tiles.txt"
    data = get_data(path)
    data_df = clean_data(data)
    data_df.to_csv("GBM_metadata.csv", index=False)

get_metadata()



