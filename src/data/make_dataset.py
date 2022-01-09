# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from src.utils.config import Config
from src.features import build_features
from dotenv import find_dotenv, load_dotenv
from sklearn.manifold import TSNE 
import umap
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import RobustScaler as rs
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.preprocessing import StandardScaler as sd

project_dir=Config.project_dir

def process_data():

    labels= pd.read_csv(project_dir / "data/raw/labels.csv")
    expression_data = pd.read_csv(project_dir / "data/raw/data.csv")

    #rename and Merge labels and features
    expression_data.rename({"Unnamed: 0":"sample"}, axis='columns', inplace =True) 
    labels.rename({"Unnamed: 0":"sample"}, axis='columns', inplace =True)
    labled_expression_merged = pd.merge(labels,expression_data,on="sample")

    # save 
    expression_data=expression_data.drop("sample",axis=1)
    expression_data.to_csv(project_dir/ "data/processed/expression_data_original.csv")
    labels=labels.drop("sample",axis=1)
    labels.to_csv(project_dir/ "data/processed/labels.csv")
    labled_expression_merged.to_csv(project_dir/ "data/processed/merged_expression_dataset.csv", index=True)

    """[Robust scaling ]
    Robust rescaling the expression levels of each gene, 
    applying the formula :
    rescaled = (gene_expression - median(gene_expression)) / IQR(gene_expression) where IQR stands for Inter Quartile Range.

    """
    expression_data_centered = rs().fit_transform(expression_data)
    df_expression_data_centered = pd.DataFrame(expression_data_centered,columns=expression_data.columns)
    df_expression_data_centered.to_csv(project_dir/ "data/processed/expression_data_centerted.csv")

    """[standard scaling ]
    """
    expression_data_standardized = sd().fit_transform(expression_data)
    df_expression_data_standardized = pd.DataFrame(expression_data_standardized,columns=expression_data.columns)
    df_expression_data_standardized.to_csv(project_dir/ "data/processed/expression_data_standardized.csv")


    y = labels['Class'].values
    true_labels = np.array([Config.labels_map[element] for element in y])
    df_true_labels = pd.DataFrame(true_labels,columns=["Class"])
    df_true_labels.to_csv(project_dir/ "data/processed/true_labels.csv")



    expression_level_5000_HGV , features_5000_HGV= build_features.top_k_variance(
        expression_data.values, 
        k=1000,
        names= expression_data.columns
        )

    #--------------------- data reduction -----------------------#
    pca_reducer = PCA(n_components=2)
    pca_reducer.fit(expression_data )
    pc = pca_reducer.transform(expression_data )
    X_tsne = TSNE(n_components=2).fit_transform(expression_data)

    UMAP_COMPONENTS_REDUCTION = 2
    UMAP_COMPONENTS_FEATURES = 20
    UMAP_EPOCHS = 2000

    manifold_reducer = umap.UMAP(
    n_components=UMAP_COMPONENTS_REDUCTION,
    n_neighbors=200, 
    n_epochs=UMAP_EPOCHS,
    metric='cosine',
    min_dist=0.9)
    manifold = manifold_reducer.fit_transform(expression_data)


    # saving tranformed data
    components= ["c1","c2"]
    df_PCA =pd.DataFrame(pc,columns=components)
    df_PCA.to_csv(Config.project_dir/ "data/transformed/PCA_reduction.csv")

    df_PCA =pd.DataFrame(X_tsne,columns=components)
    df_PCA.to_csv(Config.project_dir/ "data/transformed/TSNA_reduction.csv")

    df_PCA =pd.DataFrame(manifold,columns=components)
    df_PCA.to_csv(Config.project_dir/ "data/transformed/UMAP_reduction.csv")

    # saving hvg
    df_expression_level_5000_HGV =pd.DataFrame(expression_level_5000_HGV,columns=features_5000_HGV)
    df_expression_level_5000_HGV.to_csv(Config.project_dir/ "data/transformed/expression_data_HVG_1000.csv")



def get_data(data_type:str):
    """
    this function : 
        imports data   

    Args:
        data_type (str): ["original","centered","standardized"] the type of data you want to import 

    Returns:
        [tuple]: containing (the merged data , features , labels , true labels  )
    """
    
    merged_data= pd.read_csv(Config.data / f"processed/merged_expression_dataset.csv",index_col=0)
    features=pd.read_csv(Config.data / f"processed/expression_data_{data_type}.csv",index_col=0)
    labels=pd.read_csv(Config.data / f"processed/labels.csv",index_col=0)
    true_labels=pd.read_csv(Config.data / f"processed/true_labels.csv",index_col=0)

    return merged_data,features,labels,true_labels


def get_transformed_data():
    """
    this function : 
        import reduced data
        
    Args:
        
    Returns:
        [tuple]: containing (the merged data , features , labels , true labels  )
    """
    

    HGV= pd.read_csv(Config.data / f"transformed/expression_data_HVG_1000.csv",index_col=0)
    PCA=pd.read_csv(Config.data / f"transformed/PCA_reduction.csv",index_col=0)
    UMAP=pd.read_csv(Config.data / f"transformed/UMAP_reduction.csv",index_col=0)
    TSNA=pd.read_csv(Config.data / f"transformed/TSNA_reduction.csv",index_col=0)

    return HGV,PCA,UMAP,TSNA 


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
