from joblib import load
import numpy as np
from scipy import stats


class Classifier:

    def __init__(self):
        self.model = load('./Pipelines/PipelineFinal.joblib')
        self.features = [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 19, 20]

    def predict(self, target):
        names = target['NAME']
        target = np.array(target.drop(['NAME'], axis=1))[:, self.features]

        result = []
        for i in range(target.shape[0]):
            prediction = self.model[1].transform(self.model[0].transform([target[i]]))
            pre = {
                "Name": names[i],
                "model1": int(prediction[0, 0]),
                "model2": int(prediction[0, 1]),
                "model3": int(prediction[0, 2]),
                "mode": int(stats.mode(prediction[0])[0])
            }
            result.append(pre)
        return result
