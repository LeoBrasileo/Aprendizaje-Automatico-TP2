import re 
from typing import Any
from transformers import BertTokenizer
import pandas as pd
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List


TOKENIZER = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

ETIQUETAS_PUNT_FINAL = {
    " ": 0,
    "?": 1, 
    ".": 2, 
    ",": 3
}

ETIQUETAS_PUNT_INICIAL = {
    " ": 0,
    "¿": 1
}

ETIQUETAS_CAPITALIZACION = {
    "palabra_minuscula" : 0,
    "inicial_mayuscula" : 1,
    "multiples_mayusculas" : 2,
    "palabra_mayuscula" : 3
}


class DatasetBase(Dataset):
    def __init__(self, data: List[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indice):
        instancia = self.data[indice]

        token_ids = [token['token_id'] for token in instancia]
        capitalizacion = [token['capitalizacion'] for token in instancia]
        puntuacion_inicial = [token['puntuacion_inicial'] for token in instancia]
        puntuacion_final = [token['puntuacion_final'] for token in instancia]

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'capitalizacion': torch.tensor(capitalizacion, dtype=torch.long),
            'puntuacion_inicial': torch.tensor(puntuacion_inicial, dtype=torch.long),
            'puntuacion_final': torch.tensor(puntuacion_final, dtype=torch.long),
        }

def collate_fn(batch, padding_token_id=0, padding_etiqueta_id=-100):
    token_ids = [elem['token_ids'] for elem in batch]
    capitalizacion = [elem['capitalizacion'] for elem in batch]
    puntuacion_inicial = [elem['puntuacion_inicial'] for elem in batch]
    puntuacion_final = [elem['puntuacion_final'] for elem in batch]

    return {
        'token_ids': pad_sequence(token_ids, batch_first=True, padding_value=padding_token_id),
        'capitalizacion': pad_sequence(capitalizacion, batch_first=True, padding_value=padding_etiqueta_id),
        'puntuacion_inicial': pad_sequence(puntuacion_inicial, batch_first=True, padding_value=padding_etiqueta_id),
        'puntuacion_final': pad_sequence(puntuacion_final, batch_first=True, padding_value=padding_etiqueta_id),
    }

###############################################################################################################################

def limpiar_string (s) :
  vocales_con_acento_min = [chr(x) for x in [ord('á'), ord('é'), ord('í'), ord('ó'), ord('ú')]]
  vocales_con_acento_may = [chr(x) for x in [ord('Á'), ord('É'), ord('Í'), ord('Ó'), ord('Ú')]]
  allowed_characters = [chr(x) for x in range(97,123)] + vocales_con_acento_min + [chr(241)] + [chr(10), chr(32)] + [chr(x) for x in range(48,58)] # los caracteres de letras minúsculas, espacio en blanco y fin de linea y números
  characters_to_replace = [chr(x) for x in range(65,91)] + vocales_con_acento_may + [chr(209)] #las mayúsculas
  characters_to_replace_with = [chr(x) for x in range(97,123)] + vocales_con_acento_min + [chr(241)] #las minúsculas
  replace_dict = dict(zip(characters_to_replace, characters_to_replace_with))

  res = ""
  for c in s :
    if c in allowed_characters :
      res += c
    elif c in characters_to_replace :
      res += replace_dict[c]
  return res


def asignar_etiquetas_puntuacion(puntuacion_por_tokens):
    tokens_etiquetados = []
    for token in puntuacion_por_tokens:
        token_etiquetado = token.copy()
        token_etiquetado['puntuacion_inicial'] = ETIQUETAS_PUNT_INICIAL[token['puntuacion_inicial']]
        token_etiquetado['puntuacion_final'] = ETIQUETAS_PUNT_FINAL[token['puntuacion_final']]
        tokens_etiquetados.append(token_etiquetado)

    return tokens_etiquetados

def clasificacion_mayusculas(palabra):
    if palabra.islower(): 
        return ETIQUETAS_CAPITALIZACION['palabra_minuscula'] 
    
    elif palabra[0].isupper() and palabra[1:].islower(): 
        return ETIQUETAS_CAPITALIZACION['inicial_mayuscula']  
    
    elif palabra.isupper(): 
        return ETIQUETAS_CAPITALIZACION['palabra_mayuscula'] 
    
    else: 
        return ETIQUETAS_CAPITALIZACION['multiples_mayusculas']  


def generar_data_palabras(instancia, instancia_id):
    data_palabras = []
    for palabra in instancia.split():  
        palabra_sin_puntuacion = re.sub(r'[¿?¡!.,;:]', '', palabra)
        capitalizacion = clasificacion_mayusculas(palabra_sin_puntuacion)

        puntuacion_inicial = palabra[0] if palabra[0] == '¿' else " "
        puntuacion_final = palabra[-1] if palabra[-1] in ['?', ',', '.'] else " "

        data_palabras.append({
            'instancia_id' : instancia_id,
            'puntuacion_inicial' : puntuacion_inicial, 
            'puntuacion_final' : puntuacion_final,
            'capitalizacion' : capitalizacion
        })
        
    return data_palabras


def asignar_puntuacion_a_tokens(instancia_original: str, instancia_id: int, 
                                instancia_tokens: list[int], 
                                tokenizer: BertTokenizer=TOKENIZER) -> list[dict[str, Any]]:
    
    data_palabras = generar_data_palabras(instancia_original, instancia_id)
    
    palabra_actual = 0
    resultado = []

    for (i, token) in enumerate(instancia_tokens):
        clasificacion_token = {
            'instancia_id' : data_palabras[palabra_actual]['instancia_id'],
            'token': token,
            'token_id': tokenizer.convert_tokens_to_ids(token),
            'capitalizacion' : data_palabras[palabra_actual]['capitalizacion'],
            'puntuacion_inicial' : ' ', 'puntuacion_final' : ' '          # a priori, no sabemos si llevan puntuación.
        }
        
        # Si soy el primer token de la palabra, llevo la puntuación inicial.
        if (token[0] != "#"):
            clasificacion_token['puntuacion_inicial'] = data_palabras[palabra_actual]['puntuacion_inicial']
        
        # Si soy el último token de la palabra, llevo la puntuación final (y pasamos a la siguiente palabra)
        if (i == len(instancia_tokens)-1 or instancia_tokens[i+1][0] != "#"):
            clasificacion_token['puntuacion_final'] = data_palabras[palabra_actual]['puntuacion_final']
            palabra_actual+=1

        resultado.append(clasificacion_token)

    return resultado


def cargar_csv(paths: list[str]):
    """
    Forma del dataset: csv con atributos "texto_original, texto_limpio"
    Recibe una lista de paths y concatena todas las filas de cada archivo.
    """
    dfs = [pd.read_csv(path) for path in paths]
    dataset_df = pd.concat(dfs, ignore_index=True)
    dataset_df.to_csv("dataset_completo.csv", index=False)
    
    return dataset_df["texto_original"], dataset_df["texto_limpio"]


def generar_datos_etiquetados(paths: list[str], id_offset: int = 0, tokenizer: BertTokenizer=TOKENIZER):
    instancias_orig, instancias_procesadas = cargar_csv(paths)
    
    dataset = []
    for i, (instancia_orig, instancia_proc) in enumerate(zip(instancias_orig, instancias_procesadas)):
        dataset.append(
            asignar_etiquetas_puntuacion(
                asignar_puntuacion_a_tokens(instancia_original=instancia_orig, instancia_id=(id_offset + i), instancia_tokens=tokenizer.tokenize(instancia_proc))
            )
        )
    return dataset


def reconstruir_texto(data, caps_pred, punt_inic_pred, punt_fin_pred):
    frases = []

    frase = ""
    ult_instancia = 0

    for i in range(len(data)):
        token_data = data[i]
        token = token_data['token']
        instancia = token_data['instancia_id']

        if instancia != ult_instancia and ult_instancia != 0:
            frases.append(frase.strip())
            frase = ""
        ult_instancia = instancia
        
        puntuacion_inicial = punt_inic_pred[i]
        puntuacion_final = punt_fin_pred[i]
        capitalizacion = caps_pred[i]

        match capitalizacion:
            case 0:
                token = token.lower()
            case 1:
                token = token.capitalize()
            case 2:
                token = token.capitalize() # no se bien que hacer aca
            case 3:
                token = token.upper()

        if puntuacion_inicial == 1:
            token = "¿" + token

        match puntuacion_final:
            case 1:
                token = token + "?"
            case 2:
                token = token + "."
            case 3:
                token = token + ","

        if token.startswith("##"):
            token = token[2:]
        else:
            token = " " + token
                
        frase += token

    return frases