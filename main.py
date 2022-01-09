"""
This script can be used to :
-   regenerate some of figures in 'report/figures/generated/'
-   print all the accuracy scores for the diffrent methods  

More ditailled information and other figures can be found in the 
notebook folder 

"""
from matplotlib.pyplot import savefig
import numpy as np
from src.visualization import visualize
from src.utils.config import Config 
from src.data import make_dataset
from src.models import models


def main():

    print ("Please Wait , all the commands will be executed in a moment\n \
        All models accuracy scores will be displayed here\n \
        All the figures will be generated in 'reports/figures/generated'\n  ")

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

    # #-----------------------------------------------#
    # #----------------DATA reduction ----------------#
    # #-----------------------------------------------#

    
            #_************ PCA **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(PCA), 
    title='PCA Reduction - Cancer Classes applied on 5000 most Highly variable genes', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7) ,
    save_dir=Config.project_dir /f"reports/figures/generated/PCA.png"
    )
    
    print ("#________________ PCA_____________#  done !!  ")

            #_************ UMAP **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(UMAP), 
    title='UMAP Reduction - Cancer Classes', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7),
    save_dir=Config.project_dir /f"reports/figures/generated/UMAP.png"
    )
    print ("#________________ UMAP_____________#  done !!  ")

    #         #_************ T-SNE **************#

    visualize.visualize_dim_reduction(
    reduction=np.asarray(TSNA), 
    title='T-distributed stochastic neighbor embedding (t-SNE)', 
    labels=true_labels["Class"].values,
    s=40,
    figsize=(12,7) ,
    save_dir=Config.project_dir /f"reports/figures/generated/TSNE.png"
    )
    print ("#________________ T-SNE_____________#  done !!  ")


    """[summary]
    
    In this section , we will present all the models we used to predict classes 
    
    
    """


    #-----------------------------------------------#
    #----------------KNN  k-nerest neighbor --------#
    #-----------------------------------------------#

    KNN=models.KNN(expression_level,labels_array,"original data")
    KNN_sd=models.KNN(expression_level_sd,labels_array,"standardized")
    KNN_pca=models.KNN(PCA,labels_array,"PCA")
    KNN_umap=models.KNN(UMAP,labels_array,"UMAP")
    
    visualize.learning_curve (
    data=expression_level,
    labels=labels["Class"].values,
    model= KNN,
    range_=[3,20],
    title="KNN"
    )

    #-----------------------------------------------#
    #----------DT  decision tree            --------#
    #-----------------------------------------------#


    DT=models.DT(expression_level,labels_array,"original data")
    DT_sd=models.DT(expression_level_sd,labels_array,"standardized")
    DT_pca=models.DT(PCA,labels_array,"PCA")
    DT_umap=models.DT(UMAP,labels_array,"UMAP")
    
    visualize.learning_curve (
    data=expression_level,
    labels=labels["Class"].values,
    model= DT,
    range_=[3,20],
    title="DT"
    )
    

    #-----------------------------------------------#
    #----------SVM  Support Vector Machine  --------#
    #-----------------------------------------------#



    SVC=models.SVM(expression_level,labels_array,"original data")
    SVC_sd=models.SVM(expression_level_sd,labels_array,"standardized")
    SVC_pca=models.SVM(PCA,labels_array,"PCA")
    SVC_umap=models.SVM(UMAP,labels_array,"UMAP")
    
    visualize.learning_curve (
    data=expression_level,
    labels=labels["Class"].values,
    model= SVC,
    range_=[3,20],
    title="SVM"
    )



    #-----------------------------------------------#
    #----------NN  Neural network   --------#
    #-----------------------------------------------#

  
    #input data must be HGV
    #the encode has already been trained and stored in 'models/encoder.h5'
    models.DNN(
    X=HGV,
    labels_array=labels_array,
    description="Neural Network NN"
)






if __name__ == "__main__":
    main()