import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RF_Global():
    def __init__(self, n_estimators=100, random_state=98):
        self.modelo_puntuacion_inic = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.modelo_puntuacion_final = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.modelo_capitalizacion = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.y_caps_pred = []
        self.y_punt_inicial_pred = []
        self.y_punt_final_pred = []

    def datasetToMatrix(self, dataset):
        X = []
        y_caps = []
        y_punt_inicial = []
        y_punt_final = []

        length = len(dataset)
        ult_instancia = 0

        for i in range(len(dataset)):
            entry = dataset[i]
            instance_id = entry['instancia_id']

            features = {
                'token_id': entry['token_id'],
                'instance_id': instance_id,
                'is_first_token': int(instance_id != ult_instancia),
                'previous_token_id': dataset[i-1]['token_id'] if i > 0 else 0,
                'next_token_id': dataset[i+1]['token_id'] if i < length - 1 else 0
            }

            X.append(list(features.values()))
            y_caps.append(entry['capitalizacion'])
            y_punt_inicial.append(entry['puntuacion_inicial'])
            y_punt_final.append(entry['puntuacion_final'])

            ult_instancia = instance_id

        X = np.array(X)
        y_caps = np.array(y_caps)
        y_punt_inicial = np.array(y_punt_inicial)
        y_punt_final = np.array(y_punt_final)

        return X, y_caps, y_punt_inicial, y_punt_final

    def fit(self, data):
        X, y_caps, y_punt_inicial, y_punt_final = self.datasetToMatrix(data)         
        self.modelo_puntuacion_inic.fit(X, y_punt_inicial)
        self.modelo_puntuacion_final.fit(X, y_punt_final)
        self.modelo_capitalizacion.fit(X, y_caps)

    def predict(self, val_data):
        X, y_caps_true, y_punt_inic_true, y_punt_fin_true = self.datasetToMatrix(val_data)
        self.y_caps_pred = self.modelo_capitalizacion.predict(X)
        self.y_punt_inicial_pred = self.modelo_puntuacion_inic.predict(X)
        self.y_punt_final_pred = self.modelo_puntuacion_final.predict(X)

        return self.y_caps_pred, self.y_punt_inicial_pred, self.y_punt_final_pred