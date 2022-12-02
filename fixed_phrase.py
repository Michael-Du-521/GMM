import itertools
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from encapsulate_GMM import sliceByAnnotation, phrase_segmentation_annotation_list_dic, getScoreFromTheBest, \
    readFileThenSmoothing, \
    getStandardationResults, getSeparatedPhraseDic

#create empty dics for storing each audio analysed data
df_1_tempo_BPM_revise_smoothing_standardization_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
best_gmm_each_standard_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
best_GMM_means_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
Most_frequent_pharse_prediction_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}

# store music names
music_name1='schubert'
music_name2='etude'

#store filenames
fileName1="C:\\Users\\13104\\Desktop\\D_W_X_Schubert_D783_no15_p01_p22_95_beats_per_column_0829.csv"
fileName2="C:\\Users\\13104\\Desktop\\D_W_X_Chopin_op10_no3_p01-p22 - 40 beats per column - 0922.csv"

# remind operater which file is being processed now, and this variable name will be used in the program. Please make sure manually that the filename is consist with the music name on the process.
currently_processed_music_name= music_name1

# tempo calculation and then smoothing
df_1_tempo_BPM_revise_smoothing_list=readFileThenSmoothing(fileName=fileName1)

#standardization of smoothed tempo
df_1_tempo_BPM_revise_smoothing_standardization_dic =getStandardationResults(df_1_tempo_BPM_revise_smoothing_list)

#To get the dic containing every separated phrases of the standardation method (before:list contains dataframes /after: list contains lists)
phrases_segmentation_merged_dic=getSeparatedPhraseDic(df_1_tempo_BPM_revise_smoothing_standardization_dic,currently_processed_music_name)

#draw GMM model selection figure
fig,axes=plt.subplots(2,2)  # 2 row, 2 column empty figure

for j, k in enumerate(df_1_tempo_BPM_revise_smoothing_standardization_dic.keys()):
    #GMM fitting Process"""
    X = np.array(phrases_segmentation_merged_dic[k])
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 6)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            #Fit a Gaussian mixture with EM"""
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    best_gmm_each_standard_dic[k].append(best_gmm)
    # print('the best gmm of the standard method: '+k+'=>',best_gmm)
    bic = np.array(bic) # bic[] list to bic array
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    bars = []

    # Plot the BIC scores in the form of bar
    bar_width=0.2
    if((j+1)==1):
        spl=axes[0][0]
    elif((j+1)==2):
        spl = axes[0][1]
    elif ((j + 1) == 3):
        spl = axes[1][0]
    elif((j + 1) == 4):
        spl = axes[1][1]
    for index, (cv_type, color) in enumerate(zip(cv_types, color_iter)): # zip() function takes iterables and return a tuple
        xpos = np.array(n_components_range) + bar_width * (index - 2) # concatenate two arrays using + operator/ add an double to each elements of array
        bars.append(
            spl.bar(
                xpos,
                bic[index * len(n_components_range) : (index + 1) * len(n_components_range)],
                width=bar_width,
                color=color,
            )# return a bar-container-object
        ) # append the bar-object into a bars list
    spl.set_xticks(n_components_range) #[1,2,3,4,5]
    spl.set_ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])# set the y-limits of the current axes.
    spl.set_title("BIC score model"+' after '+k)
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))# argmin() return indices of the minimum values along the given axis.
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
        - 0.1
    )
    spl.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14) # * is a text-symbol to indicate the best model and cov type
    spl.set_xlabel("Number of components")
    spl.legend([bar[0] for bar in bars], cv_types)
print(best_gmm_each_standard_dic)
plt.show()

# to draw the results predicted by the best GMM
fig1,axes1=plt.subplots(2,2)
"""under each standard method, use the best gmm to predict"""
for i,(k, clf_list) in enumerate(best_gmm_each_standard_dic.items()):
    print('standard method:'+k)
    Y_ = clf_list[-1].predict(np.array(phrases_segmentation_merged_dic[k]))
    best_GMM_means_dic[k].append(clf_list[-1].means_)
    Y_separate=np.array_split(Y_,22)
    print(Y_separate)
    if ((i + 1) == 1):
        spl1 = axes1[0][0]
    elif ((i + 1) == 2):
        spl1 = axes1[0][1]
    elif ((i + 1) == 3):
        spl1 = axes1[1][0]
    elif ((i + 1) == 4):
        spl1 = axes1[1][1]
    # Visualize the results of the GMM prediction"""
    Pianist_index= [i for i in range(1,23)]
    plt.style.use("seaborn")
    heat_map = sns.heatmap(Y_separate, linewidth=1, annot=True, yticklabels=Pianist_index, xticklabels=[i for i in range(1, len(phrase_segmentation_annotation_list_dic[currently_processed_music_name]) + 1)], ax=spl1)
    heat_map.invert_yaxis()
    spl1.set_xlabel('Prediction_phrase')
    spl1.set_ylabel('Pianist_index')
    spl1.set_title("HeatMap:Prediction_phrase of "+k)

    # do a summary about the most frequent clustering results
    Y_=np.reshape(Y_, (22,len(phrase_segmentation_annotation_list_dic[currently_processed_music_name])))
    Y_=Y_.T
    for i in range(len(phrase_segmentation_annotation_list_dic[currently_processed_music_name])):
        Most_frequent_pharse_prediction_dic[k].append(np.bincount(Y_[i]).argmax()) #To find the most frequent predict for each phrase in every pianist's performance
    Y_=np.array(Most_frequent_pharse_prediction_dic[k]*22).reshape(22, len(phrase_segmentation_annotation_list_dic[currently_processed_music_name]))
    Most_frequent_pharse_prediction_dic[k][:]=Y_
fig1.suptitle("The heatmap of prediction of phrases: "+currently_processed_music_name)
# print('the best gmm means_:\n', best_GMM_means_dic)
plt.show()

#Visualize the most frequent phrase prediction results map
fig2,axes2=plt.subplots(2,2)
"""under each standard method, use the best gmm to predict"""
for i,k in enumerate(best_gmm_each_standard_dic.keys()):
    if ((i + 1) == 1):
        spl2 = axes2[0][0]
    elif ((i + 1) == 2):
        spl2 = axes2[0][1]
    elif ((i + 1) == 3):
        spl2 = axes2[1][0]
    elif ((i + 1) == 4):
        spl2 = axes2[1][1]
    """Visualize the results of the GMM prediction"""
    Pianist_index= [i for i in range(1,23)]
    plt.style.use("seaborn")
    heat_map = sns.heatmap(Most_frequent_pharse_prediction_dic[k], linewidth=1, annot=True, yticklabels=Pianist_index, xticklabels=[i for i in range(1, len(phrase_segmentation_annotation_list_dic[currently_processed_music_name]) + 1)], ax=spl2)
    heat_map.invert_yaxis()
    spl2.set_xlabel('Prediction_phrase')
    spl2.set_title("HeatMap:Prediction_phrase of "+k)
fig2.suptitle("The most frequent phrase prediction heat map of "+currently_processed_music_name)
plt.show()
print()

# clustering evaluation
getScoreFromTheBest(currently_processed_music_name,best_gmm_each_standard_dic,phrases_segmentation_merged_dic,Most_frequent_pharse_prediction_dic)
