import re 
from typing import Any
from transformers import BertTokenizer

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