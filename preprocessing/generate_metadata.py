import pandas as pd
import re


def get_data(path_to_file):
    data = pd.read_csv(path_to_file, names=['path'])
    return data

def clean_data(data_df):
    data_df[['root', 'tile']] = data_df['path'].str.split('/', expand=True)
    data_df['wsi']= data_df.iloc[:, 2].str.slice(0, 23)
    data_df['TCGA_barcode'] = data_df.iloc[:, 2].str.slice(0, 15)
    data_df['line_color'] = data_df["tile"].apply(lambda row: re.search('aLC_(.*?)_tn', row).group(1) if re.search('aLC_(.*?)_tn', row) else "")
    data_df["type"] = "IDHmut"
    print(data_df["type"])
    data_df = data_df.drop(columns=["root"])
    return data_df

def get_metadata(type, fold):
    path = "{}_fold{}_final.txt".format(type,fold)
    data = get_data(path)
    data_df = clean_data(data)
    data_df.to_csv("GBM_{}_fold{}_metadata.csv".format(type,fold), index=False)

get_metadata(type="IDHmut", fold=1)



