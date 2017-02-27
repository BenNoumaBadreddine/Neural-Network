import numpy as np
import time
import utilitaires7 as utilitaires

train_x= np.loadtxt("train_x.txt")
train_y= np.loadtxt("train_y.txt")
test_y= np.loadtxt("test_y.txt")
test_x= np.loadtxt("test_x.txt")
valid_y= np.loadtxt("valid_y.txt")
valid_x= np.loadtxt("valid_x.txt")

def networkPerformance(RX,test_inputs,labels_test):

        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        import pylab as pl
        expected = labels_test
        t1 = time.clock()
        predicted = RX.compute_predictions(test_inputs)
        t2 = time.clock()
        print 'Ca nous a pris ', t2-t1, ' secondes pour calculer les predictions sur ', test_inputs.shape[0],' points de test'
        err = 1.0 - np.mean(labels_test==predicted)
        print "L'erreur de test est de ", 100.0 * err,"%"
        np.save('erreurTest_cae_full',100.0 * err)

        print "Classification report for classifier %s:\n%s\n" % (
               RX, classification_report(expected, predicted))
        cm=confusion_matrix(expected, predicted)
        print "Confusion matrix:\n%s" % cm
        # Show confusion matrix in a separate window
        pl.matshow(cm)
        pl.title('Confusion matrix')
        pl.colorbar()
        pl.ylabel('True label')
        pl.xlabel('Predicted label')
        pl.show()
        return


train_x = train_x.astype(np.int32)
train_y = train_y.astype(np.int32)
test_x = test_x.astype(np.int32)
test_y = test_y.astype(np.int32)
valid_x = valid_x.astype(np.int32)
valid_y = valid_y.astype(np.int32)

n_in = train_x.shape[1]
n_classes = np.unique(train_y).shape[0]

modele = utilitaires.FeedForwardNeuralNet(n_in, n_hids=[50,50], n_out=n_classes, non_linearities="sigmoid",l2=0.001,l1=0.0001)

modele.train(train_data=train_x, train_labels=train_y, learning_rate=0.0001, batch_size=100, max_epoch=1500,monitoring_data={"ensemble de validation": (valid_x, valid_y)})

#modele.train(train_data=train_x, train_labels=train_y, learning_rate=0.001, max_epoch=3000, batch_size=10)

print modele.compute_cost(test_x, test_y)
print networkPerformance(modele,test_x,test_y)
utilitaires.plot_training_curves(modele.epochs, modele.loss_curves, title=u"Courbe d'apprentissage d'un\n reseau feedforward - Fonction de perte", ylabel="Perte")
#utilitaires.plot_training_curves(modele.epochs, modele.cost_curves, title=u"Courbe d'apprentissage d'un\n reseau feedforward - Erreur de classification", ylabel="Taux d'erreur")


#/Users/badr/anaconda/bin/python "/Users/badr/Google Drive/automne 2015/fittdist/CLASSIFICATION/RESEAU_NEURONES.PY"
 #99% : époque 1488 : perte = 0.320155 0.00288888888889
#Ca nous a pris  0.023722  secondes pour calculer les predictions sur  4500  points de test
#L'erreur de test est de  0.288888888889 %
#Classification report for classifier <utilitaires7.FeedForwardNeuralNet object at 0x107c4e890>:
 #            precision    recall  f1-score   support
#
#          0       1.00      0.99      1.00      1500
  #        1       0.99      1.00      1.00      1500
  #        2       1.00      1.00      1.00      1500
#
#avg / total       1.00      1.00      1.00      4500


#Confusion matrix:
#[[1490   10    0]
 #[   3 1497    0]
 #[   0    0 1500]]
#/Users/badr/anaconda/lib/python2.7/site-packages/matplotlib/collections.py:548: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
 # if self._edgecolors == 'face':
