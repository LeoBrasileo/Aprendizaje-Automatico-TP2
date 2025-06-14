from transformers import BertTokenizer, BertModel

def get_multilingual_token_embedding(token: str, tokenizer: BertTokenizer, model: BertModel):
    """
    Devuelve el embedding (estático) para el token.
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        print(f"❌ El token '{token}' no pertenece al vocabulario de multilingual BERT.")
        return None
    embedding_vector = model.embeddings.word_embeddings.weight[token_id]
    print(f"✅ Token: '{token}' | ID: {token_id}")
    print(f"Embedding shape: {embedding_vector.shape}")
    return embedding_vector
