import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RF_Global():
    def __init__(self, n_estimators=100, random_state=98, window_size=3):
        self.window_size = max(window_size, 1)
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
                'is_first_token': int(instance_id != ult_instancia)
            }
            for j in range(1, self.window_size + 1):
                if i - j >= 0:
                    features[f'previous_token_{j}'] = dataset[i - j]['token_id']
                else:
                    features[f'previous_token_{j}'] = 0

                if i + j < length:
                    features[f'next_token_{j}'] = dataset[i + j]['token_id']
                else:
                    features[f'next_token_{j}'] = 0

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
    
    def score(self, val_data):
        X, y_caps_true, y_punt_inic_true, y_punt_fin_true = self.datasetToMatrix(val_data)
        
        y_caps_pred = self.modelo_capitalizacion.predict(X)
        y_punt_inic_pred = self.modelo_puntuacion_inic.predict(X)
        y_punt_fin_pred = self.modelo_puntuacion_final.predict(X)

        # ignoramos las clases con 0s porque llevan a un accuracy engañoso.
        mask_caps = y_caps_true != 0
        mask_punt_inic = y_punt_inic_true != 0
        mask_punt_fin = y_punt_fin_true != 0

        score_caps = accuracy_score(y_caps_true[mask_caps], y_caps_pred[mask_caps]) if np.any(mask_caps) else None
        score_punt_inic = accuracy_score(y_punt_inic_true[mask_punt_inic], y_punt_inic_pred[mask_punt_inic]) if np.any(mask_punt_inic) else None
        score_punt_fin = accuracy_score(y_punt_fin_true[mask_punt_fin], y_punt_fin_pred[mask_punt_fin]) if np.any(mask_punt_fin) else None

        return score_caps, score_punt_inic, score_punt_fin