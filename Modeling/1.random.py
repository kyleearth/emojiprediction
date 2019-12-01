import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import time


train = pd.read_csv("./Data/train_set.csv")
#train.dropna().to_csv("./Data/train_set.csv")  # 106693

validation = pd.read_csv("./Data/validation_set.csv")
#validation.dropna().to_csv("./Data/validation_set.csv")  # 26671

test = pd.read_csv("./Data/test_set.csv")
#test.dropna().to_csv("./Data/test_set.csv")  # 14820

train = train.drop(['Unnamed: 0'],axis = 1)
validation = validation.drop(['Unnamed: 0'],axis = 1)
test = test.drop(['Unnamed: 0'],axis = 1)


x_train, y_train = train.iloc[:,0:4], train.iloc[:,5]
x_validation, y_validation = validation.iloc[:,0:4], validation.iloc[:,5]
x_test, y_test = test.iloc[:,0:4], test.iloc[:,5]


# ============= Useful Function ========================
def multiclass_AUC_plot(y_pred, y_true, plot_name, classes = [0, 1, 2, 3, 4, 5, 7, 8, 11, 13, 14, 15, 16, 17, 19]):
    '''
    Plot for AUC curves
    :param y_true:
    :param y_pred:
    :param classes:
    :return:
    '''

    y_true = label_binarize(y_true, classes)
    y_pred = label_binarize(y_pred, classes)

    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    lw = 1
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_name)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(plot_name,  dpi=600)

def recall_at_k(y_pred, y_true, k):
    ct = 0
    for i in range(len(y_true)):
        if y_true[i] in y_pred[i][0:k]:
            ct += 1
    return ct/len(y_true)
# ==========================================================

# 1. Discrete Random
print('==== Random Discrete Model- Baseline 1===')
#a) Trainning
start = time.time()
np.random.seed(0)
elements = [0,1,2,3,4,5,7,8,11,13,14,15,16,17,19]
probabilities = (y_train.value_counts()/x_train.shape[0]).values
end = time.time()
print('Time Spent: ', end-start)

#b) Validation
print('Validation Set: ')
y_pred = np.random.choice(elements, x_validation.shape[0], p=probabilities)
print('Macro f1 score %.4f' % f1_score(y_validation, y_pred, average ='macro'))
print('Micro f1 score %.4f' % f1_score(y_validation, y_pred, average ='micro'))
#multiclass_AUC_plot(y_pred, y_validation, "Discrete Random AUC - Validation")

def return_3_emojis(y_pred):
    emoji_list = [[]] * len(y_pred)
    for i in range(len(y_pred)):
        sublist = [y_pred[i]]

        # adding second emoji, has to be different
        flag = True
        while flag:
            num = int(np.random.choice(elements, 1, p=probabilities))
            if num not in sublist:
                sublist.append(num)
                flag = False
        # adding second emoji, has to be different
        flag = True
        while flag:
            num = int(np.random.choice(elements, 1, p=probabilities))
            if num not in sublist:
                sublist.append(num)
                flag = False
        emoji_list[i] = sublist
    return emoji_list

# recall @ 2, 3
pred_emojis_list = return_3_emojis(y_pred)
print("Recall at 1: ", recall_at_k(pred_emojis_list, y_validation,k = 1))
print("Recall at 2: ", recall_at_k(pred_emojis_list, y_validation,k = 2))
print("Recall at 3: ", recall_at_k(pred_emojis_list, y_validation,k = 3))


#c) Test
print('\ntest Set: ')
y_pred = np.random.choice(elements, x_test.shape[0], p=probabilities)
print('Macro f1 score %.4f' % f1_score(y_test, y_pred, average ='macro'))
print('Micro f1 score %.4f' % f1_score(y_test, y_pred, average ='micro'))
#multiclass_AUC_plot(y_pred, y_validation, "Discrete Random AUC - Test")
pred_emojis_list = return_3_emojis(y_pred)
print("Recall at 1: ", recall_at_k(pred_emojis_list, y_test,k = 1))
print("Recall at 2: ", recall_at_k(pred_emojis_list, y_test,k = 2))
print("Recall at 3: ", recall_at_k(pred_emojis_list, y_test,k = 3))
