from IPython.display import Image
import pydotplus
from sklearn import tree
# mode = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold  # 交叉验证所需的函数
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import datasets  # Built-in data set

def loadDataset(filename):
    dataset = pd.read_csv(filename,header=None)
    return dataset

if __name__ == '__main__':
    #filename = 'spam.csv'
    filename = 'test.csv'
    dataset=loadDataset(filename)
    df = pd.DataFrame(dataset, dtype='float')
    x=dataset.iloc[:,0:(df.shape[1]-1)]
    y=dataset.iloc[:,(df.shape[1]-1)]
    #dt = datasets.load_breast_cancer()  # Load data set
    print('Sample set size: ', x.shape, y.shape)
    print("---------------------------------------------------")
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=0)  # Cross validation divides training set and test set. Test_size is the proportion of test set
    list_accuracy=[]
    kf = KFold(n_splits=10)
    flag=0
    for train, test in kf.split(dataset):
        flag += 1
        print(flag)
        print("K fold division: %s %s" % (train.shape, test.shape))
        X=pd.DataFrame()
        Y=pd.DataFrame()
        for i in range(len(train)):
            X=X.append(dataset.iloc[train[i],:],ignore_index=True)
        for i in range(len(test)):
            Y=Y.append(dataset.iloc[test[i],:],ignore_index=True)
        X_train=X.iloc[:,0:(df.shape[1]-1)]
        X_test=Y.iloc[:,0:(df.shape[1]-1)]
        y_train=X.iloc[:,(df.shape[1]-1)]
        y_test=Y.iloc[:,(df.shape[1]-1)]
        print('Training set size: ', X_train.shape, y_train.shape)
        print('Test set size: ', X_test.shape, y_test.shape)
        mode = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_features=1)
        #mode = tree.DecisionTreeClassifier(criterion='entropy',splitter='best')  # use entropy/gini for feature selection
        #mode = tree.DecisionTreeClassifier(splitter='random', max_features=1)
        #mode = tree.DecisionTreeClassifier(criterion='gini',splitter='best')  # use entropy/gini for feature selection
        mode.fit(X_train, y_train)  # Use the training set to train the model
        score=mode.score(X_test, y_test)
        result=mode.predict(X_train)
        print('training set: ', accuracy_score(result,y_train))
        print('Accuracy: ', score)  # Calculate the measurement of test set (accuracy)
        list_accuracy.append(score)
        print("---------------------------------------------------")
        dot_data = tree.export_graphviz(mode, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("test"+str(flag)+".pdf")
    print('Average Error rate: ', 1-np.mean(list_accuracy))
    # for spam data entropy 0.84 without entropy 0.77