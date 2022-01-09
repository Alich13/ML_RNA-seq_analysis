import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas as pd 
import numpy as np
from scipy.sparse.construct import random 
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score , confusion_matrix ,ConfusionMatrixDisplay
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import train_test_split

from src.utils.config import Config



def get_number_of_clusters(data:np.array):
    """
    generates a plot that display the wcss score with respect to cluster number.

    we determine the number of clusters we’d like to keep. 
    To that effect, we use the Elbow-method.
    The approach consists of looking for a kink or elbow in the WCSS graph.
    Usually, the part of the graph before the elbow would be steeply declining, while the part after it – much smoother. In this instance, the kink comes at the 4 clusters mark. So, 
    we’ll be keeping a four-cluster solution.

    Args:
        data (np.array): the data we want to cluster 

    
    """

    WCSS =[]
    for i in range (1,10):
        kmean_=KMeans(n_clusters=i,init="k-means++", random_state=42)
        kmean_.fit(data)
        WCSS.append(kmean_.inertia_)
    ## plot
    plt.figure(figsize=(12,7))
    plt.plot(range(1,10),WCSS,marker='o',linestyle='--')
    plt.xlabel("number of clusters")
    plt.ylabel("WCSS")
    plt.title("")
    plt.show()
    plt.savefig(Config.project_dir /"reports/figures/generated/wcss.png")






def random_k_samples_expression_dist(X:np.array ,k :int):
    """[summary]

    Args:
        X (pd.DataFrame): expression level dataset
        k (int): randomly picked sample size  
    """

   
    
    np.random.seed(seed=7) # Set seed so we will get consistent results
    # Randomly select k samples
    samples_index = np.random.choice(range(X.shape[0]), size=k, replace=False)
    expression_levels_subset = X[samples_index,:]


    # Bar plot of expression counts by individual
    fig, ax = plt.subplots(figsize=(12, 7))
    with plt.style.context("ggplot"):
        ax.boxplot(expression_levels_subset.transpose())
        ax.set_xlabel("samples")
        ax.set_ylabel("Gene expression levels")
        ax.set_title(f"gene exression levels distributions among {k} randomly picked samples " ,fontsize=18)
        #reduce_xaxis_labels(ax, 5)
    plt.savefig(Config.project_dir /"reports/figures/generated/random_k_samples_expression_dist.png")





# Some custom x-axis labelling to make our plots easier to read
def reduce_xaxis_labels(ax, factor):
    """Show only every ith label to prevent crowding on x-axis
        e.g. factor = 2 would plot every second x-axis label,
        starting at the first.

    Parameters
    ----------
    ax : matplotlib plot axis to be adjusted
    factor : int, factor to reduce the number of x-axis labels by
    """
    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    for label in ax.xaxis.get_ticklabels()[factor-1::factor]:
        label.set_visible(True)



def visualize_dim_reduction(reduction, title, outliers_loc=None, labels=None,
                            figsize=(10, 10), save_dir=None, **kwargs):
    """Utility function for visualizing the data in a lower dimensional space.
    No matter the number of components chosen
    the function will plot only the first 2.
    Args:
        - reduction(numpy array): result of dimensionality reduction.
        - title(string): title for the plot
        - outliers_loc(iterable): index of outlying samples
        - labels(iterable): labels associated to each sample
        - **kwargs: keyword arguments passed to plt.scatter()
    Returns:
        - None
    """

    plt.figure(figsize=figsize)
    cdict = { 0: 'red', 1: 'blue', 2: 'green' , 3 :'brown',4 :'black'}
    # if we have labels

    if labels is not None:
        unique_labels = np.unique(labels).flatten()

        for i,unique_label in enumerate(unique_labels):

            indices = np.argwhere(labels == unique_label).flatten()
            plt.scatter(
                reduction[indices, 0],
                reduction[indices, 1],
                label=unique_label,
                c= cdict[i],
                ** kwargs
            )
    else:
        plt.scatter(
            reduction[:, 0],
            reduction[:, 1],
            ** kwargs
        )
    # if we know where the outliers are
    if outliers_loc is not None:

        for loc in outliers_loc:

            plt.scatter(
                reduction[loc, 0],
                reduction[loc, 1],
                c='b',
                ** kwargs
            )
            plt.annotate(
                loc,
                (reduction[loc, 0], reduction[loc, 1])
            )

    plt.xlabel(f'Component 1')
    plt.ylabel(f'Component 2')
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(
            f'{save_dir}\\{title}.png'
        )
        plt.close()
    else:
        plt.show()
        plt.close()

    plt.savefig(Config.project_dir /f"reports/figures/generated/{title}.png")

    return None



def visualize_2_subplots(reduction :np.array ,labels_1,labels_2,title,
                            figsize=(5, 10), save_dir=None, **kwargs):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize)
    ax1.set(xlabel='pc1', ylabel='pc2')
    ax2.set(xlabel='pc1', ylabel='pc2')
   
    
    fig.suptitle('PCA labled projection VS PCA kmean segment projection ')

    #we crated two colors maps for visualization purposes two make the comparison easier
    # (to have the same colors for the underlying  colors) 
    cdict_l1 = { 0: 'red', 1: 'blue', 2: 'green' , 3 :'brown',4 :'black'}
    cdict_l2 = { 0: 'blue', 1: 'red', 2: 'green' , 3 :'black',4 :'brown'}
  
 
    unique_labels = np.unique(labels_1).flatten()

    for i,unique_label in enumerate(unique_labels):

        indices = np.argwhere(labels_1 == unique_label).flatten()
        ax1.scatter(
            reduction[indices, 0],
            reduction[indices, 1],
            label=unique_label,
            c= cdict_l1[i],
            ** kwargs
        )
    
    ax1.legend(loc='upper right')

    
    unique_labels = np.unique(labels_2).flatten()
    
    for i,unique_label in enumerate(unique_labels):

        indices = np.argwhere(labels_2 == unique_label).flatten()
        ax2.scatter(
            reduction[indices, 0],
            reduction[indices, 1],
            label=unique_label,
            c= cdict_l2[i],
            ** kwargs
        )

    ax2.legend(loc='upper right')
    plt.savefig(Config.project_dir /"reports/figures/generated/kmeanVSpca.png")

    return None



# not used 
def heatmap_fig(df: pd.DataFrame, outfile: Path, color_scale: str):
    """
    Create a heatmap.

    :param df: List of percentage of features regulated by a factor in \
    a given spatial cluster
    :param outfile: The name of the figure to produce
    :param contrast: (int) the value of the contrast
    :param color_scale: The name of the color scale
    """
    data_array = df.values
    labelsx = list(df.columns)
    labelsy = list(df.index)
    index_side_dic = {l: i for i, l in enumerate(labelsy)}
    index_up_dic = {l: i for i, l in enumerate(labelsx)}
    data_up = data_array.transpose()
    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(data_up, orientation='bottom', labels=labelsx,
                               linkagefun=lambda x: linkage(x, "complete"))
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(data_array, orientation='right',
                                       labels=labelsy,
                                       linkagefun=lambda x: linkage(x, "complete"))
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)


    # Create Heatmap
    dendro_side_leaves = dendro_side['layout']['yaxis']['ticktext']
    fig['layout']['yaxis']['ticktext'] = dendro_side['layout']['yaxis']['ticktext']
    index_side = [index_side_dic[l] for l in dendro_side_leaves]
    dendro_up_leaves = fig['layout']['xaxis']['ticktext']
    heat_data = data_array[index_side, :]
    index_up = [index_up_dic[l] for l in dendro_up_leaves]
    heat_data = heat_data[:, index_up]

    if color_scale == "Picnic":
        heatmap = [
            go.Heatmap(
                x=dendro_up_leaves,
                #y=dendro_side_leaves,
                z=heat_data,
                colorbar={"x": -0.05},
                colorscale=color_scale,
                zmid=0
            )
        ]
    else:
        heatmap = [
            go.Heatmap(
                x=dendro_up_leaves,
                y=dendro_side_leaves,
                z=heat_data,
                colorbar={"x": -0.05},
                colorscale=color_scale
            )
        ]
    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    fig['layout']['yaxis']['tickvals'] = dendro_side['layout']['yaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    #
    # # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig['layout'].update({"autosize": True, "height": 1080, "width": 1920,
                             'showlegend': False, 'hovermode': 'closest',
                             })
    # Edit xaxis
    fig['layout']['xaxis'].update({'domain': [0.15, 0.8],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': True,
                                      'ticks': ""})
    # Edit xaxis2
    fig['layout'].update({'xaxis2': {'domain': [0, .15],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})

    # Edit yaxis
    fig['layout']['yaxis'].update({'domain': [0.11, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': True,
                                      'ticks': "",
                                      "side": "right"})
    # Edit yaxis2
    fig['layout'].update({'yaxis2': {'domain': [.825, 1],
                                        'mirror': False,
                                        'showgrid': False,
                                        'showline': False,
                                        'zeroline': False,
                                        'showticklabels': False,
                                        'ticks': ""}})
    plotly.offline.plot(fig, filename=str(outfile), auto_open=False)



import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools


    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(30, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(Config.project_dir /f"reports/figures/generated/{title}.png")


def learning_curve(data:pd.DataFrame,labels:np.array,model,range_:list ,title=" "):
    """[summary]

    Args:
        data (pd.DataFrame): [description]
        labels (np.array): [description]
        model ([type]): [description]
        range_ (list): [description]
    """
    training_accuracy =[]
    testing_accuracy =[]
    feature_range=range(range_[0],range_[1])
    for i in feature_range:
        Y = labels
        X = data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=Y ,test_size=0.33, random_state=42)
        model = model
        model.fit(X_train,y_train)
        training_accuracy.append(accuracy_score(model.predict(X_train),y_train))
        testing_accuracy.append(accuracy_score(model.predict(X_test),y_test))
    t = feature_range
    a = training_accuracy
    b = testing_accuracy
    plt.plot(t, a, 'r') # plotting t, a separately
    plt.plot(t, b, 'b') # plotting t, b separately
    plt.show()
    plt.savefig(Config.project_dir /f"reports/figures/generated/learning_curve{title}.png")