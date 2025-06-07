import re 

def clasificacion_mayusculas(palabra):
    if palabra.islower(): # todo en minúsculas
        return 0  
    elif palabra[0].isupper() and palabra[1:].islower(): # solo la primera letra en mayúscula
        return 1  
    elif palabra.isupper(): # todo en mayúsculas
        return 3 
    else: # mezcla de mayúsculas y minúsculas
        return 2  

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

def clasificacion_nivel_token(tokens, data_palabras, tokenizer):
    palabra_actual = 0
    resultado = []

    for (i, token) in enumerate(tokens):
        clasificacion_token = {
            'instancia_id' : data_palabras[palabra_actual]['instancia_id'],
            'token': token,
            'token_id': tokenizer.convert_tokens_to_ids(token),
            'capitalizacion' : data_palabras[palabra_actual]['capitalizacion'],
            'puntuacion_inicial' : '', 'puntuacion_final' : ''          # a priori, no sabemos si llevan puntuación.
        }
        
        # Si soy el primer token de la palabra, llevo la puntuación inicial.
        if (token[0] != "#"):
            clasificacion_token['puntuacion_inicial'] = data_palabras[palabra_actual]['puntuacion_inicial']
        
        # Si soy el último token de la palabra, llevo la puntuación final (y pasamos a la siguiente palabra)
        if (i == len(tokens)-1 or tokens[i+1][0] != "#"):
            clasificacion_token['puntuacion_final'] = data_palabras[palabra_actual]['puntuacion_final']
            palabra_actual+=1

        resultado.append(clasificacion_token)

    return resultado
