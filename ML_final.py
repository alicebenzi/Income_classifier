__author__ = 'alicebenziger'

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score , recall_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.utils import resample
from scipy.sparse import coo_matrix
from chooseFeature import chooseFeature
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
import argparse



def label_encoder(data, binary_cols, categorical_cols):
    label_enc = LabelEncoder()
    for col in categorical_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_categorical = np.array(data[categorical_cols])

    for col in binary_cols:
        label_enc.fit(data[col])
        data[col] = label_enc.transform(data[col])
    encoded_binary = np.array(data[binary_cols])
    return encoded_categorical, encoded_binary


def dummy_encoder(train_X,test_X,categorical_variable_list):
    enc = OneHotEncoder(categorical_features=categorical_variable_list)
    train_X = enc.fit_transform(train_X).toarray()
    test_X = enc.transform(test_X).toarray()
    return train_X, test_X


def resampling_sklearn(X,y):
    X_sparse = coo_matrix(X)
    X, X_sparse, y = resample(X, X_sparse, y, random_state=0)
    return X, y


def resampling(X,y,n):
    positive_indexes = [i for i, item in enumerate(y) if item == 1]
    negative_indexes = [i for i, item in enumerate(y) if item == 0]
    neg = np.random.choice(negative_indexes, n, replace=False)
    pos = np.random.choice(positive_indexes, n, replace=True)
    X_pos = X[pos,]
    X_neg = X[neg,]
    X_resamp = np.concatenate((X_pos,X_neg))
    y_pos = y[pos]
    y_neg =y[neg]
    y_resamp = np.concatenate((y_pos,y_neg))
    return X_resamp, y_resamp


def confusion_matrix(pred_values, actual_values):
    compare = [i for i, j in zip(pred_values, actual_values) if i == j]
    accuracy = float(len(compare))/len(pred_values)
    return accuracy


def scale(train_X,test_X):
    min_max_scaler= preprocessing.MinMaxScaler()
    train_X= min_max_scaler.fit_transform(train_X)
    test_X= min_max_scaler.transform(test_X)
    return train_X, test_X


def normalize(X):
    normalizer = preprocessing.Normalizer().fit(X)
    normalized_X = normalizer.transform(X)
    return normalized_X


def roc_curve_plot(true_y, pred_prob_y):
    fpr, tpr, thresholds = roc_curve(true_y, pred_prob_y[:, 1])
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


def decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(compute_importances=None, criterion='entropy',
            max_depth=5, max_features=None, max_leaf_nodes=None,
            min_density=None, min_samples_leaf=1, min_samples_split=2,
            random_state=None, splitter='best')
    clf.fit(X_train, y_train)
    plot_validation_curve(clf,title ="Decision Tree validation curve", X=X_train, y=y_train,param_name='max_depth', param_range= [3,4,5,10])
    plot_learning_curve(clf, title ="Decision Tree learning curve", X=X_train, y=y_train, ylim=(.5, 1.1))
    y_pred_DT = clf.predict(X_train)
    cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "Decision Tree : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_DT)
    file = open("DT_clf.p", "wb")
    pickle.dump(clf, file)
    file.close()


def KNearestNeighbors(X_train, y_train):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    plot_validation_curve(clf,title ="KNearest Neighbors validation curve", X=X_train, y=y_train,param_name='n_neighbors', param_range = [5, 10, 50 ,100, 500])

    plot_learning_curve(clf, title ="KNearest Neighbors learning curve", X = X_train,y = y_train, ylim=(.5, 1.1))
    y_pred_KNN = clf.predict(X_train)
    cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "KNN : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_KNN)
    file = open("KNN_clf.p", "wb")
    pickle.dump(clf, file)
    file.close()

def random_forest(X_train, y_train):
    clf = RandomForestClassifier(bootstrap=True, compute_importances=None,
            criterion='entropy', max_depth=None, max_features='log2',
            max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
            min_samples_split=2, n_estimators=40, n_jobs=1,
            oob_score=False, random_state=None, verbose=0)
    clf.fit(X_train,y_train)
    plot_validation_curve(clf,title ="Random Forest validation curve", X=X_train, y=y_train,param_name='n_estimators', param_range = [5, 10, 50 ,100, 500])

    plot_learning_curve(clf, title ="Random Forest learning curve", X = X_train,y = y_train, ylim=(.5, 1.1))

    y_pred_RF = clf.predict(X_train)
    cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "Random Forest : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_RF)
    file = open("RF_clf.p", "wb")
    pickle.dump(clf, file)
    file.close()



def naive_bayes(X_train, y_train):
    gnb=GaussianNB()
    gnb.fit(X_train,y_train)
    plot_learning_curve(gnb, title ="Naive Bayes learning curve", X = X_train,y = y_train, ylim=(0, 1.1))
    y_pred_NB = gnb.predict(X_train)
    cross_validation_accuracy= cross_val_score(gnb, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "Naive Bayes : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_NB)
    file = open("NB_clf_MN.p", "wb")
    pickle.dump(gnb, file)
    file.close()


def choose_feature(X_train, y_train):
    clf = chooseFeature()
    clf.fit(X_train,y_train)
    plot_learning_curve(clf, title ="Choose Feature learning curve", X = X_train,y = y_train, ylim=(0, 1.1))
    y_pred_CF = clf.predict(X_train)
    cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "Choose Feature : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_CF)
    file = open("CF_clf.p", "wb")
    pickle.dump(clf, file)
    file.close()


def support_vector_classifier(X_train, y_train):
    clf = SVC(kernel='rbf', C= 20, gamma= 5)
    clf.fit(X_train, y_train)
    plot_learning_curve(clf, title ="Support Vector learning curve", X = X_train,y = y_train, ylim=(0, 1.1))
    y_pred_SVC = clf.predict(X_train)
    plot_validation_curve(clf,title ="Support Vector Classifier validation curve", X=X_train, y=y_train,param_name='C', param_range = [1, 5, 20, 50])

    cross_validation_accuracy= cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy').mean()
    print "Support Vector Classifier : Training set metrics"
    print "Cross validation accuracy:", cross_validation_accuracy
    print_metrics(y_train, y_pred_SVC)
    file = open("SVM_clf.p", "wb")
    pickle.dump(clf, file)
    file.close()


def print_metrics(true_y, pred_y):
    print "Accuracy:", accuracy_score(true_y, pred_y)
    print "Precision:", precision_score(true_y, pred_y)
    print "Recall :", recall_score(true_y, pred_y)
    print "F1 score:", f1_score(true_y, pred_y)
    print "__________________________________________"


def grid_search(X_train_grid, y_train_grid, X_test, y_test, hyper_params, scores, clf):
    for score in scores:
        clf_grid = GridSearchCV(clf, hyper_params, cv=5, scoring=score)
        clf_grid.fit(X_train_grid, y_train_grid)
        print "Best estimator:", clf_grid.best_estimator_
        print "Grid search scores:"
        for params, mean_score, scores in clf_grid.grid_scores_:
                print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)
        y_pred = clf_grid.predict(X_test)

        if score == "f1":
            print "F1 on test data", f1_score(y_test, y_pred)
            print "Best estimator's cross validation score", clf_grid.best_score_

        if score == "accuracy":
            print "Accuracy on test data", accuracy_score(y_test, y_pred)
            print "Best estimator's cross validation score", clf_grid.best_score_
        if score == "precision":
            print "Accuracy on test data", precision_score(y_test, y_pred)
            print "Best estimator's cross validation score", clf_grid.best_score_


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")


    train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring = "f1")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring="f1", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="Training or Test(input must be train or test)", type=str)

    args = parser.parse_args()
    task_names = ['train', 'test']


    train = pd.read_csv('train_modified.csv',header=0)
    test = pd.read_csv('test_modified.csv',header=0)

    # Feature Selection
    del train["instance_weights"], train["industry_code"], train["occupation_code"], train["education_level"], \
        train["household_details"], train["migr_msa"], train["migr_inter_reg"], train["migr_reg"], \
        train["migr_from_sunbelt"], train["vet_qn"], train["vet_benefits"], train["year"]
    del test["instance_weights"], test["industry_code"], test["occupation_code"], test["education_level"], \
        test["household_details"], test["migr_msa"], test["migr_inter_reg"], test["migr_reg"], test["migr_from_sunbelt"], \
        test["vet_qn"], test["vet_benefits"], test["year"]

    col_names = list(train.columns.values)

    target = ['income']
    binary_variables = ['sex']
    categorical_variables = ['type_employer', 'education','marital_status', 'industry', 'occupation', 'race', 'hispanic',
                             'labor_union', 'unemp_reason', 'employment_status',  'tax_filer_status', 'region','state',
                             'household_summary', 'same_home_yr', 'parents_present', 'father_bcountry', 'mother_bcountry',
                             'self_bcountry', 'citizenship', 'self_emp']

    # all the numeric variables (those not in binary or categorical variable columns)
    numeric_variables = [col for col in col_names if col not in binary_variables if col not in categorical_variables]

    encoded_categorical, encoded_binary = label_encoder(train, binary_variables, categorical_variables)
    encoded_categorical_test, encoded_binary_test= label_encoder(test, binary_variables, categorical_variables)

    numerical_data = np.array(train[numeric_variables])
    numerical_data_test = np.array(test[numeric_variables])

    training_data_transformed = np.concatenate((encoded_categorical,encoded_binary,numerical_data),axis=1)
    test_data_transformed = np.concatenate((encoded_categorical_test,encoded_binary_test,numerical_data_test),axis=1)

    train_X = training_data_transformed[:, :-1]
    train_y = training_data_transformed[:, -1]
    test_X = test_data_transformed[:, :-1]
    test_y = test_data_transformed[:, -1]

    #dummy encoding
    train_X, test_X = dummy_encoder(train_X, test_X, categorical_variable_list = list(range(0,20,1)))

    #normalizing
    train_X = normalize(train_X)
    test_X = normalize(test_X)

    #balancing data - bootstrapping the minority class
    #print np.unique(train_y,return_counts=True)
    X_train_resamp, y_train_resamp = resampling(train_X, train_y, n = 90000)
    #print np.unique(y_train_resamp,return_counts=True)


    if args.task_name == 'train':

        ## Training###
        decision_tree(X_train_resamp, y_train_resamp)
        random_forest(X_train_resamp, y_train_resamp)
        naive_bayes(X_train_resamp, y_train_resamp)
        choose_feature(X_train_resamp, y_train_resamp)


        #resampling with a smaller training set for SVM and KNN
        X_train_resamp, y_train_resamp = resampling(train_X, train_y, n = 20000)
        KNearestNeighbors(X_train_resamp, y_train_resamp)
        support_vector_classifier(X_train_resamp, y_train_resamp)




    else:
        ##Testing####
        #Decision Tree###
        file = open("DT_clf.p", "rb")
        dt_clf = pickle.load(file)
        file.close()
        dt_pred = dt_clf. predict(test_X)
        print "Decision Tree : Test set metrics"
        print_metrics(test_y, dt_pred)
        #pred_prob_DT = dt_clf.predict_proba(test_X)
        #roc_curve_plot(test_y,pred_prob_DT)


        ###Random Forest####
        file = open("RF_clf.p", "rb")
        rf_clf = pickle.load(file)
        rf_pred = rf_clf.predict(test_X)
        print "Random Forest : Test set metrics"
        print_metrics(test_y, rf_pred)
        file.close()
        # pred_prob_RF = rf_clf.predict_proba(test_X)
        # roc_curve_plot(test_y,pred_prob_RF)



        ##Choose Feature####
        file = open("CF_clf.p", "rb")
        cf_clf = pickle.load(file)
        cf_pred = cf_clf.predict(test_X)
        print "Choose Feature : Test set metrics"
        print_metrics(test_y, cf_pred)
        file.close()
        # pred_prob_CF = cf_clf.predict_proba(test_X)
        # roc_curve_plot(test_y,pred_prob_CF)


        # ###Naive Bayes####
        file = open("NB_clf.p", "rb")
        nb_clf = pickle.load(file)
        nb_pred = nb_clf.predict(test_X)
        print "Naive Bayes : Test set metrics"
        print_metrics(test_y, nb_pred)
        file.close()
        # pred_prob_NB = nb_clf.predict_proba(test_X)
        # roc_curve_plot(test_y,pred_prob_NB)
        #

        ##Support Vector Machines####
        file = open("SVM_clf.p", "rb")
        svm_clf = pickle.load(file)
        svm_pred = svm_clf.predict(test_X)
        print "Support Vector Classifier : Test set metrics"
        print_metrics(test_y, svm_pred)
        file.close()
        # pred_prob_SVM = svm_clf.predict_proba(test_X)
        # roc_curve_plot(test_y,pred_prob_SVM)



        # ###KNNt####
        # file = open("KNN_clf.p", "rb")
        # knn_clf = pickle.load(file)
        # knn_pred = knn_clf.predict(test_X)
        # print "KNN : Test set metrics"
        # print_metrics(test_y, knn_pred)
        # file.close()
        # pred_prob_KNN = knn_clf.predict_proba(test_X)
        # roc_curve_plot(test_y,pred_prob_KNN)









