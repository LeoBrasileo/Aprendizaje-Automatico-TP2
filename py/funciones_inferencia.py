import torch
import pandas as pd
import csv
from collections import defaultdict

def transformar_csv_a_instancias(path: str):
    """ 
    Toma el path de un csv como el dado por la cátedra, de columnas
        instancia_id,token_id,token
    Y devuelve un diccionario como el que toma la función de predicción
    """
    resultado = {
        'instancia_id': [],
        'token_id': [],
        'token': []
    }
    with open(path, 'r', encoding='utf-8') as f:
        lector = csv.DictReader(f)
        for fila in lector:
            resultado['instancia_id'].append(fila['instancia_id'])
            resultado['token_id'].append(fila['token_id'])
            resultado['token'].append(fila['token'])
    return resultado


def predecir_para_tokens(model, inputs: dict[str, list[int | str]]):
    """
    inputs es un diccionario con las keys y valores:
            instancia_id: lista con todas las instancia_id,
            token_id: lista con todos los token_id,
            token: lista con todos los tokens
    """
    # Agrupamos por instancia los tokens y token_ids
    inputs_por_instancia = defaultdict(lambda: {'token_ids': [], 'tokens': []})
    for input in inputs:
        instancia_ids = input['instancia_id']
        token_ids = input['token_id']
        tokens = input['token']
        for inst_id, token_id, token in zip(instancia_ids, token_ids, tokens):
            inputs_por_instancia[inst_id]['token_ids'].append(token_id)
            inputs_por_instancia[inst_id]['tokens'].append(token)

    model.eval()

    # Para cada instancia, pasamos sus token_ids por el modelo y guardamos las predicciones de cada uno de sus tokens en la lista de resultados
    resultados = []
    for instancia_id, datos in inputs_por_instancia.items():
        token_ids = datos['token_ids']
        tokens = datos['tokens']

        tensor_token_ids = torch.tensor([token_ids], dtype=torch.long)  

        with torch.no_grad():
            logits_punt_inic, logits_punt_final, logits_capitalizacion = model(tensor_token_ids)

        pred_punt_inic = torch.argmax(logits_punt_inic, dim=-1).squeeze(0).tolist()
        pred_punt_final = torch.argmax(logits_punt_final, dim=-1).squeeze(0).tolist()
        pred_capitalizacion = torch.argmax(logits_capitalizacion, dim=-1).squeeze(0).tolist()

        for token_id, token, punt_inic, punt_final, capitalizacion in zip(token_ids, tokens, pred_punt_inic, pred_punt_final, pred_capitalizacion):
            resultados.append({
                "instancia_id": instancia_id,
                "token_id": token_id,
                "token": token,
                "puntuacion_inicial": punt_inic,
                "puntuacion_final": punt_final,
                "capitalizacion": capitalizacion
            })

    return resultados