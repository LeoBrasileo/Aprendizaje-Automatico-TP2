import requests
import re
import os

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

def save_cleaned_text(text, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Cleaned text saved to: {output_path}")
        return True
    except IOError as e:
        print(f"Error al guardar: {e}")
        return False

if __name__ == "__main__":
    output_file = "marquez_cleaned.txt"
    raw_text = load_text_from_github()
    
    if raw_text:
        cleaned_text = clean_text(raw_text)
        print(f"Primeros 200: {cleaned_text[:200]}...")
        
        if save_cleaned_text(cleaned_text, output_file):
            print(f"Output file: {os.path.abspath(output_file)}")
    else:
        print("Failed to load text from GitHub")