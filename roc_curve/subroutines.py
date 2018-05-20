# These are the subroutines that will be used by the Jupyter notebook
import random as rd
import numpy as np

class ROC:


    def __init__(self,mu_neg,sig_neg,mu_pos,sig_pos):
        """
        :param mu_neg: The mean of the negative samples
        :param sig_neg: The stdev of the negative samples
        :param mu_pos: The mean of the positive samples
        :param sig_pos: The stdev of the positive samples
        """
        self.mu_neg = mu_neg
        self.mu_pos = mu_pos
        self.sig_neg = sig_neg
        self.sig_pos = sig_pos

    def generate_samples(self, Nsamp_pos,Nsamp_neg):
        """
        :param Nsamp_pos: True positive samples
        :param Nsamp_neg: True Negative samples
        :return: Specified number of positive and negative samples
        """

        # Generate N1 true positive and N2 true negative samples
        s_Tpos = np.random.normal(self.mu_pos, self.sig_pos, Nsamp_pos)
        s_Tneg = np.random.normal(self.mu_neg, self.sig_neg, Nsamp_neg)

        # Now we label the above samples appropriately
        # 0 = False
        # 1 = True
        X_Tpos = np.asarray([[s_Tpos[i],1] for i in range(len(s_Tpos))])
        X_Tneg = np.asarray([[s_Tneg[i],0] for i in range(len(s_Tneg))])

        # Append both arrays
        X_result = np.append(X_Tpos,X_Tneg,axis=0)

        # Shuffle the array
        np.random.shuffle(X_result)


        return X_result



    def classify(self,data_vec,t):
        """
        This function classifies all objects with x>t as positive(=1) samples
        :param t: The threshhold parameter
        :return: The labels for all of the points as predicted by Threshold parameter
        """

        # The decision function
        def decision(x):
            d=0.0

            if(x>=t):
                d=1.0

            return d

        labels = np.asarray([decision(data_vec[i]) for i in range(len(data_vec))])

        return labels



    def confusion_matrix(self,vec_data,vec_true_labels,vec_classifier_labels):
        """
        Calculate the Confusion Matrix for the Model

        :param vec_data:
        :param vec_true_labels:
        :param vec_classifier_labels:
        :return:
        m11: True Positives
        m12: False Negatives
        m21: False Positives
        m22: True Negatives
        """

        # The number of samples
        Nsamples = len(vec_data)

        # Count the number of true positives
        TP = 0.0 # True positives
        TN = 0.0 # True Negatives
        FP = 0.0 # False Positives
        FN = 0.0 # False Negatives

        for i in range(Nsamples):

            if(vec_true_labels[i]==1 and vec_classifier_labels[i]==1):
                TP = TP+1.0

            if(vec_true_labels[i]==0 and vec_classifier_labels[i]==0):
                TN = TN +1.0

            if(vec_true_labels[i]==0 and vec_classifier_labels[i]==1):
                FP = FP+1.0

            if(vec_true_labels[i]==1 and vec_classifier_labels[i]==0):
                FN = FN+1.0

        m11 = TP
        m22 = TN
        m12 = FN
        m21 = FP

        return m11,m12,m21,m22

    def classification_metrics(self,m11,m12,m21,m22):
        """
        This function uses the confusion Matrix to calculate classifier metrics

        :param m11: True Positives
        :param m12: False Negatives
        :param m21: False Positives
        :param m22: True Negatives
        :return: accuracy,precision,recall,F1_score
        """

        accuracy = (m11+m22)/(m11+m12+m21+m22)

        if(m11+m21==0.0):
            precision = 0.0
        else:
            precision = m11/(m11+m21)

        if(m11+m12==0.0):
            recall=0.0
        else:
            recall = m11/(m11+m12)

        if(precision+recall==0.0):
            F1_score = 0.0
        else:
            F1_score = (2.0*recall*precision)/(recall+precision)

        return accuracy,precision,recall,F1_score

    def ROC_curve(self,vec_samples,vec_sample_labels,tmin,tmax,dt):
        """

        This function returns the

        :param tmin:
        :param tmax:
        :param dt:
        :return:
        """

        ROC_array=[]

        #1. Classiy the data
        for t in np.arange(tmin,tmax,dt):
            vec_classifier_labels = self.classify(vec_samples, t)
            m11, m12, m21, m22 = self.confusion_matrix(vec_samples, vec_sample_labels, vec_classifier_labels)
            acc,prec,rec,F1 = self.classification_metrics(m11, m12, m21, m22)
            ROC_array.append([t,m11,m21,acc,prec,rec,F1])


        return np.asarray(ROC_array)

