"""
Author: Iason Skylitsis
CV Lab Group 1
Email: cs151097@uniwa.gr
AM: 151097

"""
import cv2
import os
from sklearn.cluster import MiniBatchKMeans  # KMeans
from sklearn.cluster import AffinityPropagation
import numpy as np
# import secondary functions that will be used very frequent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


# important: this code has been tested using the following opencv and supportive libraries versions.
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16

# -----------------------------------------------------------------------------------------
# ---------------- SUPPORTING FUNCTIONS GO HERE -------------------------------------------
# -----------------------------------------------------------------------------------------

# return a dictionary that holds all images category by category.
def load_images_from_folder(folder, inputImageSize):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat)
            # print(' .. parsing image', cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize it, if necessary
                img = cv2.resize(img, (inputImageSize[0], inputImageSize[1]))

                category.append(img)
        images[filename] = category
        print(' . Finished parsing images. What is next?')
    return images


# Creates descriptors using an approach of your choise. e.g. ORB, SIFT, SURF, FREAK, MOPS, ετψ
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features(images, detector):
    print(' . start detecting points and calculating features for a given image set')
    detector_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    if detector == 'sift':
        detectorToUse = cv2.xfeatures2d.SIFT_create()
    elif detector == 'surf':
        detectorToUse = cv2.xfeatures2d.SURF_create()
    else:
        detectorToUse = cv2.ORB_create()
    for nameOfCategory, availableImages in images.items():
        features = []
        for img in availableImages:  # reminder: val
            kp, des = detectorToUse.detectAndCompute(img, None)
            descriptor_list.extend(des)
            features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set')
    return [descriptor_list, detector_vectors]  # be aware of the []! this is ONE output as a list


# A k-means clustering algorithm who takes 2 parameter which is number
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeansVisualWordsCreation(k, descriptor_list):
    print(' . calculating central points for the existing feature values.')
    # kmeansModel = KMeans(n_clusters = k, n_init=10)
    batchSize = np.ceil(descriptor_list.__len__() / 50).astype('int')
    kmeansModel = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, verbose=0)
    kmeansModel.fit(descriptor_list)
    visualWords = kmeansModel.cluster_centers_  # a.k.a. centers of reference
    print(' . done calculating central points for the given feature set.')
    return visualWords, kmeansModel


def affinityVisualWordsCreation(descriptor_list):
    print(' . calculating central points for the existing feature values.')
    # bandwidth = estimate_bandwidth(noisy_flat, quantile=0.3, n_samples=200)
    affinityModel = AffinityPropagation()
    affinityModel.fit(descriptor_list)
    visualWords = affinityModel.cluster_centers_  # a.k.a. centers of reference
    print(' . done calculating central points for the given feature set.')
    return visualWords, affinityModel


# Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
# image. For each keypoints in an image, we will find the nearest center, defined using training set
# and increase by one its value
def mapFeatureValsToHistogram(DataFeaturesByClass, visualWords, TrainedKmeansModel):
    # depenting on the approach you may not need to use all inputs
    histogramsList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]

    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues:  # yes, we check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedKmeansModel.predict(tmpImageFeatures))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # do not forget to normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram / numberOfDetectedPointsInThisImage

            # now update the input and output coresponding lists
            histogramsList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    return histogramsList, targetClassList


def display(dir):
    directory = os.listdir(dir)
    for each in directory:
        plt.figure()
        currentFolder = dir + '/' + each
        for i, file in enumerate(os.listdir(currentFolder)[0:6]):
            fullpath = currentFolder + "/" + file
            img = mpimg.imread(fullpath)
            plt.subplot(2, 3, i + 1)
            plt.suptitle(each + 's')
            plt.axis('off')
            plt.imshow(img)
        plt.savefig('OutputData/' + each + '.png')


# here we run the code
import time

start_time = time.time()

# define a fixed image size to work with
inputImageSize = [200, 200, 3]  # define the FIXED size that CNN will have as input

# define the path to train and test files

TrainImagesFilePath = 'C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_250_80-20/train'
TestImagesFilePath = 'C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_250_80-20/test'

# Show some pictures from each category in the dataset
# The images are saved in the OutputData folder
display(TrainImagesFilePath)

# Create a list of the feature extraction techniques we are going to use
detector_list = ['sift', 'surf', 'orb']

# Create a list of the dataset folders we are going to use
datasets_list = ['C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_250_80-20',
                 'C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_250_60-40',
                 'C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_15_80-20',
                 'C:/Users/Jason/Desktop/Datasets/kagglecatsanddogs_3367a_15_60-40'
                 ]

# create dataframe with the data that will be written in excel
df = pd.DataFrame(columns=['Feature Extraction', 'Clustering Technique', 'Classification Technique',
                                    'Train Data Ratio', 'Accuracy (tr)', 'Precision (tr)',
                                    'Recall(tr)', 'F1 score (tr)', ' Accuracy (te)', 'Precision (te)',
                                    ' Recall(te)', 'F1 score (te)'])
#Loop for all datasets in the dataset list
for dataset in datasets_list:
    #Loop for all the detectors in the detector list
    for detector in detector_list:
        # define the path to train and test files
        TrainImagesFilePath = dataset + '/train'
        TestImagesFilePath = dataset + '/test'

        # get the train data ratio
        parts = dataset.split('_')
        ratio = parts[3]
        number_of_images = parts[2]
        # load the train images
        trainImages = load_images_from_folder(TrainImagesFilePath,
                                              inputImageSize)  # take all images category by category for train set

        # calculate points and descriptor values per image
        trainDataFeatures = detector_features(trainImages, detector)
        # Takes the descriptor list which is unordered one
        TrainDescriptorList = trainDataFeatures[0]

        # create the central points for the histograms using k means.
        # here we use a rule of the thumb to create the expected number of cluster centers
        numberOfClasses = trainImages.__len__()  # retrieve num of classes from dictionary
        possibleNumOfCentersToUse = 10 * numberOfClasses
        if number_of_images == '250':
            visualWords, TrainedKmeansModel = kmeansVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList)
            clustering_technique = 'k-means'
        elif number_of_images == '15':
            clustering_technique = 'Affinity Propagation'
            visualWords, TrainedKmeansModel = affinityVisualWordsCreation(TrainDescriptorList)
        # Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
        trainBoVWFeatureVals = trainDataFeatures[1]

        # create the train input train output format
        trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords,
                                                                          TrainedKmeansModel)
        # X_train = np.asarray(trainHistogramsList)
        # X_train = np.concatenate(trainHistogramsList, axis=0)
        X_train = np.stack(trainHistogramsList, axis=0)

        # Convert Categorical Data For Scikit-Learn
        from sklearn import preprocessing

        # Create a label (category) encoder object
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(trainTargetsList)
        # convert the categories from strings to names
        y_train = labelEncoder.transform(trainTargetsList)

        # train and evaluate the classifiers
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier().fit(X_train, y_train)
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))

        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))

        from sklearn.svm import SVC

        svm = SVC()
        svm.fit(X_train, y_train)
        print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))

        # ----------------------------------------------------------------------------------------
        # now run the same things on the test data.
        # DO NOT FORGET: you use the same visual words, created using training set.

        # clear some space
        del trainImages, trainBoVWFeatureVals, trainDataFeatures, TrainDescriptorList

        # load the train images
        testImages = load_images_from_folder(TestImagesFilePath,
                                             inputImageSize)  # take all images category by category for train set

        # calculate points and descriptor values per image
        testDataFeatures = detector_features(testImages, detector)

        # Takes the sift feature values that is seperated class by class for train data, we need this to calculate the histograms
        testBoVWFeatureVals = testDataFeatures[1]

        # create the test input / test output format
        testHistogramsList, testTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords,
                                                                        TrainedKmeansModel)
        X_test = np.array(testHistogramsList)
        y_test = labelEncoder.transform(testTargetsList)

        # classification tree
        # predict outcomes for test data and calculate the test scores
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('')
        print(' Printing performance scores:')
        print('')

        print('Accuracy scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of Decision Tree classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')

        # create dataframe with the data that will be written in excel
        df2 = pd.DataFrame([[detector, clustering_technique, 'Decision Trees', ratio, acc_train.round(2), pre_train.round(2), rec_train.round(2),
                             f1_train.round(2), acc_test.round(2), pre_test.round(2), rec_test.round(2), f1_test.round(2)]],
                           columns=['Feature Extraction', 'Clustering Technique', 'Classification Technique',
                                    'Train Data Ratio', 'Accuracy (tr)', 'Precision (tr)',
                                    'Recall(tr)', 'F1 score (tr)', ' Accuracy (te)', 'Precision (te)',
                                    ' Recall(te)', 'F1 score (te)'])
        df = df.append(df2, ignore_index=True)

        # support vector machines
        # now check for both train and test data, how well the model learned the patterns
        y_pred_train = gnb.predict(X_train)
        y_pred_test = gnb.predict(X_test)
        # calculate the scores
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        pre_train = precision_score(y_train, y_pred_train, average='macro')
        pre_test = precision_score(y_test, y_pred_test, average='macro')
        rec_train = recall_score(y_train, y_pred_train, average='macro')
        rec_test = recall_score(y_test, y_pred_test, average='macro')
        f1_train = f1_score(y_train, y_pred_train, average='macro')
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        # print the scores
        print('Accuracy scores of SVM classifier are:',
              'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
        print('Precision scores of SVM classifier are:',
              'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
        print('Recall scores of SVM classifier are:',
              'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
        print('F1 scores of SVM classifier are:',
              'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
        print('')
        # create dataframe with the data that will be written in excel
        df2 = pd.DataFrame([[detector, clustering_technique, 'SVM', ratio, acc_train.round(2), pre_train.round(2), rec_train.round(2),
                             f1_train.round(2), acc_test.round(2), pre_test.round(2), rec_test.round(2), f1_test.round(2)]],
                           columns=['Feature Extraction', 'Clustering Technique', 'Classification Technique',
                                    'Train Data Ratio', 'Accuracy (tr)', 'Precision (tr)',
                                    'Recall(tr)', 'F1 score (tr)', ' Accuracy (te)', 'Precision (te)',
                                    ' Recall(te)', 'F1 score (te)'])
        df = df.append(df2, ignore_index=True)

# write in excel
try:
    df.to_excel('OutputData/results.xlsx')
except:
    print(' Unable to write to file')
print('End of program.py')
print("--- %s seconds ---" % (time.time() - start_time))
