from joblib import load
import numpy as np

class Classifier:

    def __init__(self):
        self.model1 = load('./Pipelines/modelExample.joblib')
        self.model2 = load('./Pipelines/modelExample.joblib')
        self.model3 = load('./Pipelines/modelExample.joblib')

    def predict(self, target):
        nombres=target['Nombre']
        target=np.array(target.drop(['Nombre'],axis=1))

        result=[]
        for i in range (target.shape[0]):

            pre={
                "nombre":nombres[i],
                "model1":int(self.model1.predict([np.array(target)[i]])),
                "model2":int(self.model2.predict([np.array(target)[i]])),
                "model3":int(self.model3.predict([np.array(target)[i]]))
            }
            result.append(pre)
        return result



