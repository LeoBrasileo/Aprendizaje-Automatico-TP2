import random
import requests
import re
import os
import csv
import torch
from torch.utils.data import Dataset
import torch.nn as nn


characters = [ #incluye variantes de los nombres 
    "José Arcadio Buendía",
    "Úrsula Iguarán",
    "Úrsula",
    "Aureliano Buendía",
    "Aureliano",
    "José Arcadio",
    "Amaranta",
    "Rebeca",
    "Arcadio",
    "Santa Sofía de la Piedad",
    "Aureliano José",
    "Remedios Moscote",
    "Pilar Ternera",
    "Fernanda del Carpio",
    "Renata Remedios",
    "Remedios la Bella",
    "Aureliano Segundo",
    "José Arcadio Segundo",
    "Mauricio Babilonia",
    "Petra Cotes",
    "Gaston",
    "Melquíades",
    "Prudencio Aguilar",
    "Visitación",
    "Pietro Crespi",
    "Catarino",
    "Gerineldo Márquez",
    "Don Apolinar Moscote",
    "Nigromanta",
    "Crespi",
    "Carmela",
    "Trinidad",
    "Alfonso",
    "Aureliano Babilonia",
    "Gabriel",
    "Mercedes",
    "Virgilio",
    "Arcadio",
    "Fernanda",
    "Renata",
    "Remedios",
    "Mauricio",
    "Petra",
    "Prudencio",
    "Pietro",
    "Gerineldo",
    "Apolinar",
]

def cargar_nombres_csv(path=os.path.abspath("data/nombres.csv")):
    nombres = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                nombres.append(row[0].strip())
    return nombres

def load_text_from_github():
    """cargar el texto de un GitHub gist"""
    url = "https://gist.githubusercontent.com/ismaproco/6781d297ee65c6a707cd3c901e87ec56/raw/gabriel_garcia_marquez_cien_annos_soledad.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  
        
        # (assuming UTF-8 encoding)
        text = response.text
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"Error al pedir el texto: {e}")
        return None
    
def replace_proper_nouns(text, characters = characters):
    patrones = [re.escape(p) for p in sorted(characters, key=lambda x: -len(x.split()))]
    patron = r'\b(' + '|'.join(patrones) + r')\b'
    nombres_comunes = cargar_nombres_csv()

    def reemplazo(match):
        return random.choice(nombres_comunes)

    return re.sub(patron, reemplazo, text)

def clean_text(text):
    """Limpia el texto eliminando estructuras específicas y caracteres no deseados"""
    if text is None:
        return None

    # Eliminar ocurrencias del patrón: número + saltos de línea + "Cien años de soledad" + saltos de línea + "Gabriel García Márquez"
    pattern = r'(?:\d+\s*)?\s*\n*\s*Cien años de soledad\s*\n+\s*Gabriel García Márquez\s*(?:\s*\d+)?'

    text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def clean_text_for_training(text):
    # elimina caracteres raros
    text = text.replace("«", "")
    text = text.replace("»", "")
    #no saco los "-" porque vienen de dialogos y me puede servir para identidicar frases de personajes
    #me arrepenti -> lo saco
    text = text.replace("-", "")
    text = text.replace("—", "")

    # saco todos los nombres propios
    text = replace_proper_nouns(text)
    return text

def save_cleaned_text(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Cleaned text saved to: {output_path}")
        return True
    except IOError as e:
        print(f"Error al guardar: {e}")
        return False
    
def has_words_full_capitalized(sentence):
    """Verifica si en una oracion una palabra está completamente en mayúsculas"""
    sentence = sentence.split()
    return any(word.isupper() and len(word) > 1 for word in sentence)

def is_decorative(sentence):
    return re.fullmatch(r'[-–—=*\s]+', sentence) or sentence.isdigit()
    
def keep_important_sentences(text):
    """El texto crudo esta lleno de oraciones que no aportan valor semantico para nuestro modelo. Esta función limpia el texto para optimizar el entrenamiento"""
    new_sentences = []
    sentences = text.split('.')
    for sentence in sentences:
        if has_words_full_capitalized(sentence):
            continue
        if is_decorative(sentence):
            continue

        new_sentences.append((sentence + ".").strip())

    return new_sentences

def sentences_to_paragraph(sentences):
    """
    Convierte una lista de oraciones en una lista de párrafos.
    Cada párrafo contiene 2 o 3 oraciones.
    """
    paragraphs = []
    i = 0
    while i < len(sentences):
        num_sentences = random.choice([2, 3])
        group = sentences[i:i+num_sentences]

        paragraph = " ".join(group).strip()
        paragraph = " ".join(paragraph.split())  # Elimina espacios extra
        paragraphs.append(paragraph.strip())

        i += num_sentences
    return paragraphs

def clean_sentence(paragraph):
    """Limpia los párrafos eliminando espacios innecesarios y normalizando el texto"""
    vocales_con_acento_min = [chr(x) for x in [ord('á'), ord('é'), ord('í'), ord('ó'), ord('ú')]]
    vocales_con_acento_may = [chr(x) for x in [ord('Á'), ord('É'), ord('Í'), ord('Ó'), ord('Ú')]]
    allowed_characters = [chr(x) for x in range(97,123)] + vocales_con_acento_min + [chr(241)] + [chr(10), chr(32)] + [chr(x) for x in range(48,58)] # los caracteres de letras minúsculas, espacio en blanco y fin de linea y números
    characters_to_replace = [chr(x) for x in range(65,91)] + vocales_con_acento_may + [chr(209)] #las mayúsculas
    characters_to_replace_with = [chr(x) for x in range(97,123)] + vocales_con_acento_min + [chr(241)] #las minúsculas
    replace_dict = dict(zip(characters_to_replace, characters_to_replace_with))

    res = ""
    for c in paragraph:
        if c in allowed_characters:
            res += c
        elif c in characters_to_replace:
            res += replace_dict[c]
    return res

def sentences_to_csv(sentences, output_csv):
    """Guarda los parrafos importantes y su parte limpia en un archivo CSV"""
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['texto_original', 'texto_limpio'])
            for sentence in sentences:
                cleaned_sentence = clean_sentence(sentence)
                writer.writerow([sentence, cleaned_sentence])
        print(f"Paragraphs saved to: {output_csv}")
        return True
    except IOError as e:
        print(f"Error al guardar el CSV: {e}")
        return False

if __name__ == "__main__":
    output_file = "data/marquez_por_oracion.txt"
    output_csv = "data/marquez_por_oracion.csv"
    raw_text = load_text_from_github()
    
    if raw_text:
        cleaned_text = clean_text(raw_text)
        #print(f"Primeros 200: {cleaned_text[:200]}...")
        
        if save_cleaned_text(cleaned_text, output_file):
            print(f"Output file: {os.path.abspath(output_file)}")

        cleaned_text = clean_text_for_training(cleaned_text)

        sentences = keep_important_sentences(cleaned_text)
        sentences_to_csv(sentences, output_csv)

    else:
        print("Failed to load text from GitHub")


# --- Dataset ---
class MarquezDataset(Dataset):
    def __init__(self, data, label_key='puntuacion_final', pad_token_id=0, pad_label_id=-100):
        """
        Args:
            data: list of sentences, each a list of dicts (one per token)
            label_key: which label to use ('puntuacion_final' or 'puntuacion_inicial')
            pad_token_id: ID used for padding tokens
            pad_label_id: label used for padding positions (-100 for CrossEntropyLoss)
        """
        self.data = data
        self.label_key = label_key
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]

        # Extract features
        token_ids = [tok['token_id'] for tok in sentence]
        caps = [tok['capitalizacion'] for tok in sentence]
        punt_inicial = [tok['puntuacion_inicial'] for tok in sentence]
        punt_final = [tok[self.label_key] for tok in sentence]

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'capitalizacion': torch.tensor(caps, dtype=torch.long),
            'puntuacion_inicial': torch.tensor(punt_inicial, dtype=torch.long),
            'puntuacion_final': torch.tensor(punt_final, dtype=torch.long),
            'length': len(token_ids)
        }

def marquez_collate_fn(batch, pad_token_id=0, pad_label_id=-100):
    # Separate each field
    token_ids = [item['token_ids'] for item in batch]
    caps = [item['capitalizacion'] for item in batch]
    punt_inicial = [item['puntuacion_inicial'] for item in batch]
    punt_final = [item['puntuacion_final'] for item in batch]
    lengths = [item['length'] for item in batch]

    # Pad sequences
    padded_token_ids = nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)
    padded_caps = nn.utils.rnn.pad_sequence(caps, batch_first=True, padding_value=0.0)
    padded_punt_inicial = nn.utils.rnn.pad_sequence(punt_inicial, batch_first=True, padding_value=pad_label_id)
    padded_punt_final = nn.utils.rnn.pad_sequence(punt_final, batch_first=True, padding_value=pad_label_id)

    return {
        'token_ids': padded_token_ids,
        'capitalizacion': padded_caps,
        'puntuacion_inicial': padded_punt_inicial,
        'puntuacion_final': padded_punt_final,
        'lengths': torch.tensor(lengths)
    }