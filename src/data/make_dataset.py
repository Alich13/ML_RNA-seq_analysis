# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from src.utils.config import Config
from dotenv import find_dotenv, load_dotenv


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
