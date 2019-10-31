'''
Author: Sumanth Paranjape
Libraries: sckit, pandas, numpy, datetime, matplotlib.
Function: Does data preprocessing, feature extraction and
          dimesionality reduction using PCA.
'''
from datetime import datetime
from datetime import timedelta
from itertools import cycle, islice
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

patientNum = None
noOfMergedCols = None

#Cleaning CGM data from CGMSeriesLunchPat data files
def cleanCGMData(success, cgmFrame):
    prevCGM = None
    while not success:
        try:
            global patientNum
            patientNum = input("Enter Patient Number:")
            cgmFrame = pd.read_csv('CGMSeriesLunchPat'+patientNum+'.csv')
            for start, row in cgmFrame.iterrows():
                for end in range(0, 31):
                    if(np.isnan(row[end]) or row[end] is None and prevCGM is not None):
                        cgmFrame.iat[start, end] = prevCGM
                    prevCGM = row[end]
            cgmFrame = cgmFrame.fillna(method='ffill')
            cgmFrame = cgmFrame.fillna(method='bfill')
            success = True;
        except Exception as e:
            print("Patient Number doesnt exist, try again!")
    return cgmFrame

#Cleaning and formatting Timestamps from CGMDatenumLunchPat data files
def cleanAndFormatTime(timeFrame, patientNum):
    timeFrame = pd.read_csv('CGMDatenumLunchPat'+patientNum+'.csv')
    columnsList = list(timeFrame.columns)
    formatedDateFrame = pd.DataFrame(index=range(0, len(timeFrame.index)), columns=columnsList)
    prevDateTime = datetime.min
    for start, row in timeFrame.iterrows():
        for end in range(0, 31):
            if(np.isnan(row[end]) or row[end] is None):
                date_time = prevDateTime-timedelta(minutes = 5)
                formatedDateFrame.iat[start, end] = date_time.strftime("%m/%d/%Y %H:%M:%S")
            else:
                formatedDateFrame.iat[start, end] = getFormattedTime(int(row[end]), row[end]).strftime("%m/%d/%Y %H:%M:%S")
            prevDateTime = date_time if(np.isnan(row[end])) else getFormattedTime(int(row[end]), row[end])
    return formatedDateFrame

#Converts dateNum to dateTime format
def getFormattedTime(dateNum, floatVal):
    return datetime.fromordinal(dateNum) + timedelta(days=floatVal%1) - timedelta(days = 366)

def mergeData(cgmFrame, formatedDatesFrame):
    global noOfMergedCols 
    noOfMergedCols = len(cgmFrame.index)*2
    mergedFrame = pd.DataFrame(index=range(0, 31), columns=np.arange(noOfMergedCols))
    idx = 0
    datesList = []
    cgmDataList = []
    for start, row in cgmFrame.iterrows():
        for end in range(0, 31):
            datesList.append(formatedDatesFrame.iat[start, end])
            cgmDataList.append(cgmFrame.iat[start, end])
        datesList.reverse()
        cgmDataList.reverse()
        #Copies list values into column idx and idx+1
        mergedFrame[idx] = list(islice(cycle(datesList), len(mergedFrame)))
        mergedFrame[idx+1] = list(islice(cycle(cgmDataList), len(mergedFrame)))
        datesList = []
        cgmDataList = []
        idx += 2
    plotGraphs(mergedFrame)
    return mergedFrame

def plotGraphs(mergedFrame):
    for index in range(0, len(mergedFrame.columns), 2):
        plt.xlabel("Time 2hours 30mins") 
        plt.ylabel("Blood Glucose mg/dL") 
        plt.plot(mergedFrame[index], mergedFrame[index+1])
        plt.savefig('plots/CGM'+str(index)+'.png')
        plt.clf()

def extractFeatures(mergedFrame):
    global noOfMergedCols
    featureMatrix = pd.DataFrame(columns=['Timestamp','Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6'])
    fftArray = None
    timeList = mergedFrame[0] 
    #print("StartTime:"+timeList[0]+",EndTime:"+timeList[30])
    for index in range(0, noOfMergedCols, 2):
        cgmList = mergedFrame[index+1]
        fftArray = fft(cgmList)
        timeStamp = mergedFrame[index][0] +' - '+mergedFrame[index][30]
        feature1 = np.var(fftArray)
        feature2 = averageCGMFoodIntake(cgmList)
        feature3 = percentHypoglycemia(cgmList)
        feature4 = percentHyperglycemia(cgmList)
        feature5 = np.std(cgmList)
        feature6 = np.mean(cgmList) 
        featureMatrix = featureMatrix.append({'Timestamp': timeStamp, 'Feature1': feature1, 'Feature2': feature2, 'Feature3': feature3, 'Feature4': feature4, 'Feature5': feature5, 'Feature6': feature6}, ignore_index=True)
    #plotFFT(fftArray)
    plotXYGraph(featureMatrix)
    return featureMatrix

def plotXYGraph(featureMatrix): 
    x = np.arange(len(featureMatrix))
    for i in range(1,7):
        y = featureMatrix['Feature'+str(i)]
        plt.plot(x, y) 
        plt.xlabel("Data Samples") 
        plt.ylabel('Feature'+str(i)) 
        plt.title('Feature'+str(i)+' graph') 
        plt.savefig('plots/Feature'+str(i)+'.png')
        #plt.show() 
        
def plotPCs(varianceResults):
    x = ['PC1','PC2', 'PC3', 'PC4', 'PC5']
    y = varianceResults
    plt.clf()
    plt.bar(x, y, align='center', alpha=1.0) 
    plt.xlabel("Principle Components") 
    plt.ylabel('Percentage Variance') 
    plt.title('Percentage Variance of PCs') 
    plt.show() 

def plotFFT(fftArray):
    size = 30
    spacing = 1.0 / 800.0
    xframe = np.linspace(0.0, 1.0/(2.0*spacing), size/2)
    fig, ax = plt.subplots()
    ax.plot(xframe, 2.0/size * np.abs(fftArray[:size//2]))
    plt.show()
        
def averageCGMFoodIntake(cgmList):
    beforeMealAvg = np.mean(cgmList[0: 7])
    afterMealAvg = np.mean(cgmList[8: len(cgmList)])
    return abs(afterMealAvg-beforeMealAvg)

def percentHypoglycemia(cgmList):
    return (len(cgmList[cgmList<=70])/len(cgmList))*100

def percentHyperglycemia(cgmList):
    return (len(cgmList[cgmList>=180])/len(cgmList))*100

def pricipalCompToFeatureMap(normalizedMatrix):
    model = PCA(n_components=5).fit(normalizedMatrix)
    X_pc = model.transform(normalizedMatrix)
    n_pcs= model.components_.shape[0]
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items())
    print("Most important features:")
    print(df)

def reduceDimensions(featureMatrix):
    features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    normalize = featureMatrix.loc[:, features].values
    normalize = StandardScaler().fit_transform(normalize)
    #print(normalize)
    pca = PCA(n_components=5)
    principleComponents = pca.fit_transform(normalize)
    reducedDF = pd.DataFrame(data = principleComponents
             , columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    finalDF = pd.concat([reducedDF, featureMatrix[['Timestamp']]], axis = 1)
    #print(abs( pca.components_ ))
    plotPCs(pca.explained_variance_ratio_*100)
    pricipalCompToFeatureMap(normalize)
    return finalDF

def plotPCAGraph(reducedDF):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_ylabel('PC3', fontsize = 15)
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='green')
    ax.tick_params(axis='z', colors='blue')
    ax.set_title('Top 3 components for comparing', fontsize = 20)
    ax.scatter(reducedDF['PC1'], reducedDF['PC2'], reducedDF['PC3'])
    plt.show()
    

def main():
    #Main controller here
    success = False
    cgmFrame = pd.DataFrame()
    timeFrame = pd.DataFrame()
    
    #Preprocesses CGM data and sets patientNum
    cgmFrame = cleanCGMData(success, cgmFrame)
    print("=========CLEANING CGM,TIMESTAMPS AND MERGING================")
    print("Forward fill and backward fill of NaN values")
    #Formats MatLab's dateNum format into dateTime format
    global patientNum
    formatedDatesFrame = cleanAndFormatTime(timeFrame, patientNum)
    #print(formatedDatesFrame)
    print("======END OF CLEANING CGM DATA=========")
    print("Plotting CGM graphs against time...graphs can be found under plots/ folder")
    print("Plotting...")
    #Merge the CGM and Time frames into one csv file
    mergedFrame = mergeData(cgmFrame, formatedDatesFrame)
    print("==========MERGED DATA FRAME============")
    print(mergedFrame)
    mergedFrame.to_csv ('Patient'+str(patientNum)+'Merged.csv', index = None, header=True)
    #mergedFrame.to_csv (r'output.csv', index = None, header=True)
    print("Merged CSV file saved as Patient"+str(patientNum)+"Merged.csv")
    print("======END OF MERGING CGM DATA=========")
    featureMatrix = extractFeatures(mergedFrame)
    print("======EXTRACTED FEATURE MATRIX========")
    print(featureMatrix)
    print("======END OF FEATURE EXTRACTION=========")
    reducedDF = reduceDimensions(featureMatrix)
    print("======FEATURE MATRIX AFTER DIMENSIONALITY REDUCTION====")
    print(reducedDF)
    plotPCAGraph(reducedDF)
    print("======END OF PRINCIPLE COMPONENT ANALYSIS=========")
    
    
if __name__ == "__main__":
    main()
