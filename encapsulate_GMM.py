import itertools

import numpy as np
import pandas as pd
from numpy import array
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score


# create empty list for storing each audio analysed data"""
phrase_segmentation_annotation_list_dic={'etude':[8, 8, 8, 8, 8], 'mozart':[24, 24, 24, 24, 24, 24, 12, 24, 24, 12], 'schubert':[12, 12, 12, 12, 12, 12, 12, 12], 'ballade':[9, 22, 24, 24, 23, 24, 24, 25, 21, 27, 22, 24]}
phrases_annotation_label_true={'etude':[0, 1, 0, 1, 2], 'mozart':[0, 0, 0, 0, 1, 0, 2, 1, 0, 2], 'schubert':[0,0,0,0,1,2,1,2], 'ballade':[0, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 5]}
# create empty list for storing each audio analysed data"""

def readFileThenSmoothing(fileName):
    """input a fixed unit score name of the csv file, and return a tempo list after smoothing """
    #Read the csv file, and convert it to the DataFrame"""
    df_1_tempo_BPM_revise_list=[]
    df_1_tempo_BPM_revise_smoothing_list = []
    df = pd.read_csv(fileName,header=None)

    # Standardize the axis name as well as the rows and column number of the DataFrame"""
    df = df.set_axis([i for i in range(1, len(df)+1)], axis='index')
    df = df.set_axis([i for i in range(1, 23)], axis='columns')
    df = df.rename_axis('beat index', axis='index')
    df = df.rename_axis('pianist', axis='columns')

    #smoothing process
    for i in range(1, 23):
        #"use for-loop to go through all the columns(each pianist) in the target Dataframe
        df_1 = df.loc[:, i]  # copy all the rows(:), but only keep the i th column of the df
        #using difference（差分）Function  of DataFrame to calculate the beat interval (and further, the tempo)
        df_1_beat_interval = df_1.diff()
        #since after differenced, each column only had len(df)-1 remained rows, so we revise the column length of the ith column data of the df
        df_1_beat_interval_revise = df_1_beat_interval.iloc[1:len(df)]
        df_1_tempo_BPM_revise_list.append(60 / (df_1_beat_interval_revise))
        #append the mean of the first len(df)-1 tempo to the original list,  to integrate a len(df) tempo dataframe"""
        df_1_tempo_BPM_revise_list[i - 1] = df_1_tempo_BPM_revise_list[i - 1].append(pd.Series([np.mean(df_1_tempo_BPM_revise_list[i - 1])]), ignore_index=True)  # append mean
        # special treatment for the schubert
        if (len(df)==95):
            df_1_tempo_BPM_revise_list[i - 1] = df_1_tempo_BPM_revise_list[i - 1].append(pd.Series([np.mean(df_1_tempo_BPM_revise_list[i - 1])]), ignore_index=True)  # append mean again only for schubert
            df_1_tempo_BPM_revise_list[i - 1] = df_1_tempo_BPM_revise_list[i - 1].set_axis([i for i in range(1, len(df) + 2)], axis='index')
        # for etude of fixed unit phrase
        else:
            df_1_tempo_BPM_revise_list[i - 1] = df_1_tempo_BPM_revise_list[i - 1].set_axis([i for i in range(1, len(df) + 1)], axis='index')
        #the process of tempo BPM smoothing

        #the smoothing window size is 3, which mean we do the averaging in a window of size 3 moving from the upmost to the end of the ith column of the DataFrame"""
        window_size = 3
        # print('type=>\n',type(df_1_tempo_BPM_revise_list[0]),'\n','<=type') # it is a Series type.
        df_1_tempo_BPM_revise_smoothing = df_1_tempo_BPM_revise_list[i - 1].rolling(window_size).mean()
        #the first smoothing value is calculated by equaling to corresponding the first beat interval value of the original position"""
        df_1_tempo_BPM_revise_smoothing[1] = df_1_tempo_BPM_revise_list[i - 1][1]
        # the second smoothing value is calculated by averaging the first-two beat interval values of original position in the df_1_tempo_BPM_revise_list
        df_1_tempo_BPM_revise_smoothing[2] = np.mean(df_1_tempo_BPM_revise_list[i - 1][0:2])
        # append the data series after smoothing into the corresponding list
        df_1_tempo_BPM_revise_smoothing_list.append(df_1_tempo_BPM_revise_smoothing)
    # print("array(df_1_tempo_BPM_revise_smoothing_list).shape:",array(df_1_tempo_BPM_revise_smoothing_list).shape)
    return df_1_tempo_BPM_revise_smoothing_list


def getStandardationResults(df_1_tempo_BPM_revise_smoothing_list):
    """input a smooth list, execute tempo standardization process, and then return a standardization dic of after each standardization method"""
    df_1_tempo_BPM_revise_smoothing_standardization_dic = {'zscore': [], 'range_regulation': [], 'mean_regulation': [],'log_scaling': []}
    for i in range(1, 23):
        #(1) standard score regulation for original tempo after smoothing"""
        df_1_tempo_BPM_revise_smoothing_zscore = (df_1_tempo_BPM_revise_smoothing_list[i - 1] - np.mean(
            df_1_tempo_BPM_revise_smoothing_list[i - 1])) / df_1_tempo_BPM_revise_smoothing_list[i - 1].std()
        df_1_tempo_BPM_revise_smoothing_standardization_dic['zscore'].append(df_1_tempo_BPM_revise_smoothing_zscore)
        #(2) range regulation for original tempo after smoothing"""
        df_1_tempo_BPM_revise_smoothing_range_regulation = (df_1_tempo_BPM_revise_smoothing_list[i - 1] - np.min(
            df_1_tempo_BPM_revise_smoothing_list[i - 1])) / (np.max(
            df_1_tempo_BPM_revise_smoothing_list[i - 1]) - np.min(df_1_tempo_BPM_revise_smoothing_list[i - 1]))
        df_1_tempo_BPM_revise_smoothing_standardization_dic['range_regulation'].append(
            df_1_tempo_BPM_revise_smoothing_range_regulation)
        #(3) mean regulation for original tempo after smoothing"""
        df_1_tempo_BPM_revise_smoothing_mean_regulation = (
                df_1_tempo_BPM_revise_smoothing_list[i - 1] / np.mean(df_1_tempo_BPM_revise_smoothing_list[i - 1]))
        df_1_tempo_BPM_revise_smoothing_standardization_dic['mean_regulation'].append(
            df_1_tempo_BPM_revise_smoothing_mean_regulation)
        #(4) log2 scaling for original tempo after smoothing"""
        df_1_tempo_BPM_revise_smoothing_log_scaling = np.log2(df_1_tempo_BPM_revise_smoothing_list[i - 1])
        df_1_tempo_BPM_revise_smoothing_standardization_dic['log_scaling'].append(
            df_1_tempo_BPM_revise_smoothing_log_scaling)
    # print('df_1_tempo_BPM_revise_smoothing_standardization_dic',df_1_tempo_BPM_revise_smoothing_standardization_dic)
    return df_1_tempo_BPM_revise_smoothing_standardization_dic

#utilized functions
#for unfixed-unit musical score(audio)
def sliceByAnnotation(slice_beats,annotation_list):
    """phrase segmentation slice by shape indicated by score annotation"""
    separated_by_beat_list=[]
    for i in annotation_list:
        temp_list=slice_beats[:i]
        slice_beats=slice_beats[i:]
        separated_by_beat_list.append(temp_list)
    return separated_by_beat_list

#for fixed-unit musical score(audio)
def getSeparatedPhraseDic(df_1_tempo_BPM_revise_smoothing_standardization_dic,currently_processed_music_name):
    phrases_segmentation_dic = {'zscore': [], 'range_regulation': [], 'mean_regulation': [], 'log_scaling': []}
    phrases_segmentation_merged_dic = {'zscore': [], 'range_regulation': [], 'mean_regulation': [], 'log_scaling': []}
    for k in df_1_tempo_BPM_revise_smoothing_standardization_dic.keys():
        for i in range(1, 23):
            df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1] =\
            df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1].tolist()
            # spilt each sublist in to len() parts(that is :len(sublist) vectors), because the etude and schubert is fixed-unit phrase audio.
            phrases_segmentation_dic[k].append(np.array_split(df_1_tempo_BPM_revise_smoothing_standardization_dic[k][i - 1],len(phrase_segmentation_annotation_list_dic[currently_processed_music_name])))
            # to create a list containing all the len(sublist)-beat elements' vectors, and store these into a dic respectively by standardization names
            phrases_segmentation_merged_dic[k] = list(itertools.chain(*phrases_segmentation_dic[k]))
    return  phrases_segmentation_merged_dic


#standardization for polynomial
# Least-squares fitting for each unequal phrase to make sure it returns a equal-length coefficients(based on the shortest beats phrase)
def backSameAmountCoefficientsFromUnfixedPhrases(nth_order,phrases_vector):
    pass



#go through all the tests
def getScoreFromTheBest(currently_processed_music_name,best_gmm_each_standard_dic,phrases_segmentation_merged_dic,Most_frequent_pharse_prediction_dic):
    """cluster performance evaluation """
    print("(1)score needs annotation prior knowledge")
    for k, clf in best_gmm_each_standard_dic.items():
        labels_true= phrases_annotation_label_true[currently_processed_music_name]
        # labels_true = phrases_annotation_label_true[currently_processed_music_name][1:]# only for ballade eliminating the first phrase
        labels_pred = Most_frequent_pharse_prediction_dic[k][-1]
        print('standardization method:' + k+f" for the {currently_processed_music_name}")
        # Rand index=> Rand index is a function that measures the similarity of the two assignments, ignoring permutations:
        print('metrics.rand_score:',metrics.rand_score(labels_true, labels_pred))
        print('metrics.adjusted_rand_score:',metrics.adjusted_rand_score(labels_pred, labels_true))
        # Mutual Information based scores=>the Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations.
        print('metrics.adjusted_mutual_info_score:',metrics.adjusted_mutual_info_score(labels_true, labels_pred))
        print('metrics.normalized_mutual_info_score',metrics.normalized_mutual_info_score(labels_true, labels_pred))
        # V-measure
        print('metrics.v_measure_score:',metrics.v_measure_score(labels_true, labels_pred))
        print()

    print("(2)score does not need annotation prior knowledge")
    for k, clf in best_gmm_each_standard_dic.items():
        try:
                print('metrics.silhouette_score:', metrics.silhouette_score(np.array(phrases_segmentation_merged_dic[k]),clf[-1].predict(np.array(phrases_segmentation_merged_dic[k])),metric='euclidean'))
        except:
            pass
        try:
                print("davies_bouldin_score:", davies_bouldin_score(np.array(phrases_segmentation_merged_dic[k]),clf[-1].predict(np.array(phrases_segmentation_merged_dic[k]))))
        except:
            pass


