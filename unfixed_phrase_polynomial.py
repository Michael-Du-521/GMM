import itertools
import math

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import array
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from coustic_research.preparation.encapsulate_GMM import sliceByAnnotation, phrase_segmentation_annotation_list_dic, \
    readFileThenSmoothing, getStandardationResults, getScoreFromTheBest

# store music names
music_name1='schubert'
music_name2='etude'
music_name3='mozart'
music_name4='ballade'

#create empty list for storing each audio analysed data"""
phrases_segmentation_dic = {'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
df_1_tempo_BPM_revise_smoothing_standardization_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
phrases_segmentation_merged_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
Most_frequent_pharse_prediction_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
After_getting_coefficients_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
best_gmm_each_standard_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
best_GMM_means_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}
Most_frequent_pharse_prediction_dic={'zscore':[],'range_regulation':[],'mean_regulation':[],'log_scaling':[]}

#store filenames
fileName1="C:\\Users\\13104\\Desktop\\D_W_X_Schubert_D783_no15_p01_p22_95_beats_per_column_0829.csv"
fileName2="C:\\Users\\13104\\Desktop\\D_W_X_Chopin_op10_no3_p01-p22 - 40 beats per column - 0922.csv"
fileName3="C:\\Users\\13104\\Desktop\\D_W_X_Mozart_K331_1st-mov_p01_p22_216_beats_per_column_0922.csv"
fileName4="C:\\Users\\13104\\Desktop\\D_W_X_Chopin_op38_p01_p22_269_beats_per_column_0912.csv"

# remind operater which file is being processed now, and this variable name will be used in the program. Please make sure manually that the filename is consist with the music name on the process.
currently_processed_music_name= music_name2

#polynomial_degree for fitting
polynomial_degree= 4 # 4 for etude  2 for schubert
# tempo calculation and then smoothing
df_1_tempo_BPM_revise_smoothing_list=readFileThenSmoothing(fileName=fileName2)

#standardization of smoothed tempo
df_1_tempo_BPM_revise_smoothing_standardization_dic =getStandardationResults(df_1_tempo_BPM_revise_smoothing_list)


#just for unfixed unit separation of phrases
for k in df_1_tempo_BPM_revise_smoothing_standardization_dic.keys():
    for i in range(1, 23):
        df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1] = sliceByAnnotation(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i-1], phrase_segmentation_annotation_list_dic[currently_processed_music_name])
        #spilt each sublist in to corresponding unequal-united parts"""
        phrases_segmentation_dic[k].append(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1])
        #to create a list containing all the elements' vectors"""
        phrases_segmentation_merged_dic[k] = list(itertools.chain(*phrases_segmentation_dic[k]))

# (1) fit the polynomial curve for each standardization method from every pianist
for k in df_1_tempo_BPM_revise_smoothing_standardization_dic.keys():
    for i in range(1,23):
        fig, axes = plt.subplots(2, int(math.ceil(len(phrase_segmentation_annotation_list_dic[currently_processed_music_name])/2)))
        for l,ax in zip(range(0,len(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1])),axes.flat):# l is the x th phrase
            X = array([j + 1 for j in range(len(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1][l]))]).reshape(-1,1)

            y = df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i-1][l]
            poly_reg = PolynomialFeatures(degree=polynomial_degree)
            X_poly = poly_reg.fit_transform(X)
            lin_reg2 = LinearRegression()
            lin_reg2.fit(X_poly, y)
            After_getting_coefficients_dic[k].append(lin_reg2.coef_)
            X_grid = np.arange(min(X), max(X), 0.1)# the dense x axe especially for the smoothed fitted polynomial curve
            X_grid = X_grid.reshape(len(X_grid), 1)
            ax.scatter(X, y, color='red')
            ax.set_xticks(X,minor=True)
            ax.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
            # ax.plot(X_grid, lin_reg2.predict(X_poly), color='blue')
            ax.set_xlabel("beat index")
            plt.suptitle(f"The smoothed and standard of {k} tempo curve of the "+str(i)+' pianist')
            plt.savefig(fname="The smooth tempo curve of the "+str(i)+' pianist after '+k+' standardization')
# plt.close('all')
        plt.show()


# # (2)fit the polynomial curve for each standardization method from every pianist( but do not draw polynomial curve) ((1) and (2) only can run one at a time )
# for k in df_1_tempo_BPM_revise_smoothing_standardization_dic.keys():
#     for i in range(1,23):
#         for l in range(0,len(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1])):
#             X = array([j + 1 for j in range(len(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1][l]))]).reshape(-1,1)
#             y = df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i-1][l]
#             poly_reg = PolynomialFeatures(degree=polynomial_degree)
#             X_poly = poly_reg.fit_transform(X)
#             lin_reg2 = LinearRegression()
#             lin_reg2.fit(X_poly, y)
#             After_getting_coefficients_dic[k].append(lin_reg2.coef_)
# print()

#draw GMM model selection figure
fig,axes=plt.subplots(2,2)  # 2 row, 2 column empty figure
for j, k in enumerate(df_1_tempo_BPM_revise_smoothing_standardization_dic.keys()):
    #GMM fitting Process"""
    X = np.array(After_getting_coefficients_dic[k])
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
plt.show()

# to draw the results predicted by the best GMM
fig1,axes1=plt.subplots(2,2)
# under each standard method, use the best gmm to predict"""
for i,(k, clf_list) in enumerate(best_gmm_each_standard_dic.items()):
    print('standard method:'+k)
    Y_ = clf_list[-1].predict(np.array(After_getting_coefficients_dic[k]))
    best_GMM_means_dic[k].append(clf_list[-1].means_)
    Y_separate=np.array_split(Y_,22)
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
getScoreFromTheBest(currently_processed_music_name,best_gmm_each_standard_dic,After_getting_coefficients_dic,Most_frequent_pharse_prediction_dic)
