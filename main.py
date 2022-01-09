"""
This script can be used to :
-   regenerate some of figures in 'report/figures/generated/'
-   print all the accuracy scores for the diffrent methods  

More ditailled information and other figures can be found in the 
notebook folder 

"""
import numpy as np
from src.visualization import visualize
from src.utils.config import Config 
from src.data import make_dataset


def main():

    #-----------------------------------------------#
    #----------------Import Data--------------------#
    #-----------------------------------------------#
    
    labled_data_set,expression_level,labels,true_labels= make_dataset.get_data("original")
    labels_array= labels["Class"].values

    print ("#________________ Data importation _____________#  done !!  ")


    #-----------------------------------------------#
    #--------Import precessed data Process Data-----#
    #-----------------------------------------------#


    labled_data_set_sd,expression_level_sd,labels,true_labels= make_dataset.get_data("standardized")
    
    HGV,PCA,UMAP,TSNA = make_dataset.get_transformed_data()

    print ("#________________ processing _____________#  done !!  ")

    #-----------------------------------------------#
    #----------------explore Data--------------------#
    #-----------------------------------------------#


    visualize.random_k_samples_expression_dist(np.asarray(expression_level, dtype=float),40)

    print ("#________________ explore data _____________#  done !!  ")

    #-----------------------------------------------#
    #----------------DATA reduction ----------------#
    #-----------------------------------------------#

    
            #_************ PCA **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(PCA), 
    title='PCA Reduction - Cancer Classes applied on 5000 most Highly variable genes', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7) )
    print ("#________________ PCA_____________#  done !!  ")

            #_************ UMAP **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(UMAP), 
    title='UMAP Reduction - Cancer Classes', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7) )
    print ("#________________ UMAP_____________#  done !!  ")

            #_************ T-SNE **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(TSNA), 
    title='T-distributed stochastic neighbor embedding (t-SNE)', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7) )
    print ("#________________ T-SNE_____________#  done !!  ")




if __name__ == "__main__":
    main()