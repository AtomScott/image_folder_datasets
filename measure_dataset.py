import glob
import pickle
import os

from joblib import Parallel, delayed
import numpy as np
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from torchvision import transforms as tfms
# from lightgbm import LGBMClassifier

import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from image_folder_datasets.core import CNNModule, ImageFolderDataModule

import pandas as pd
from contexttimer import Timer
from sklearn import metrics

def multiclass_report(x_train, y_train, x_val, y_val, clf=None, dataset_name=None):
    """Utility function to score classifier
    Pass in the classifier if you want to test train, test times etc.
    """
    n_classes = len(set(y_train))
    labels = sorted(list(set(y_train)))
    
    with Timer() as train_time:
        clf.fit(x_train, y_train)
        
    with Timer() as test_time:
        y_pred_proba = clf.predict_proba(x_val)
        
    y_pred = np.argmax(y_pred_proba, axis=1)
        
    results = {
        'Train time': train_time.elapsed,
        'Test time': test_time.elapsed
    }
    results['clf'] = clf.__class__.__name__
    results['dataset'] = dataset_name
    results['Weighted Fscore'] = metrics.f1_score(y_val, y_pred, average='weighted')
    results['Top-1 score'] = metrics.top_k_accuracy_score(y_val, y_pred_proba, k=1)
    results['Top-5 score'] = metrics.top_k_accuracy_score(y_val, y_pred_proba, k=5) if n_classes > 5 else None
    results['n_classes'] = n_classes
    results['n_train_samples'] = len(x_train)
    results['n_test_samples'] = len(x_val)
        
    return results


def parfunc(data_dir):
    try:
        print("START:", data_dir)
        results = []
        dataset_name = data_dir
        # print(dataset_name)
        dm = ImageFolderDataModule(data_dir, 256, transform, num_workers=20)
        dm.setup()
        
        x_train, y_train = zip(*[(np.asarray(x), y) for x, y in dm.trainset])
        x_val, y_val = zip(*[(np.asarray(x), y) for x, y in dm.valset])
        
        # Do dimensionality reduction to 
        pca = PCA(n_components=0.9)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
        # print("\tn_components:", pca.n_components_)
        
        svm = SVC(probability=True)
        results.append(multiclass_report(x_train, y_train, x_val, y_val, clf=svm, dataset_name=dataset_name))

        dummy_clf = DummyClassifier()
        results.append(multiclass_report(x_train, y_train, x_val, y_val, clf=dummy_clf, dataset_name=dataset_name))
        
        print("DONE:", data_dir)
        return results
    except:
        return []
    # print("\t", pd.DataFrame(results[2*i:2*i+2]))
    
    
if __name__ == "__main__":
    data_dirs = sorted(list(glob.glob('datasets/*')))

    transform = tfms.Compose([
        tfms.Grayscale(),
        tfms.Resize(128, interpolation=2),
        tfms.RandomCrop(112),
        tfms.ToTensor(),
        Rearrange('h w c -> (h w c)'), 
    ])

    print(len(data_dirs))
    result = Parallel(n_jobs=5)(delayed(parfunc)(data_dir) for data_dir in data_dirs)
    
    with open("svm_dummy.txt", "wb") as fp:
        pickle.dump(result, fp)
        
    pd.DataFrame(result).to_csv('svm_dummy_results.csv')