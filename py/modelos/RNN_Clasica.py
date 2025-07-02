import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def get_multilingual_token_embedding(token: str):
  token_id = tokenizer.convert_tokens_to_ids(token)

  if token_id is None or token_id == tokenizer.unk_token_id:
    return None

  embedding_vector = model.embeddings.word_embeddings.weight[token_id]
  return embedding_vector

class RNN_Clasica(nn.Module):
  def __init__(self, 
               hidden_size,
               embedding_dim, 
               vocab_size,
               initial_punct_classes=3,
               final_punct_classes=5,
               cap_punct_classes=5, 
               bert_embedding=True,
               dropout_rate=0.5
               ):
    
    super(RNN_Clasica, self).__init__()
    
    # Si tenemos embedding pre-entrenados, los usamos
    if bert_embedding:
      #self.embedding = get_multilingual_token_embedding

      self.embedding = nn.Embedding.from_pretrained(model.embeddings.word_embeddings.weight, freeze=True)

    else:
      self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    # Dos capas LSTM 
    self.RNN = nn.RNN(embedding_dim, hidden_size, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=False)
    
    # Dropout
    # Capas de output para las tres tareas que queremos
    self.RNN_puntuacion_inic = nn.RNN(hidden_size, initial_punct_classes, batch_first=True, dropout=dropout_rate, bidirectional=False)     # Sin puntuación || ¿
    self.RNN_puntuacion_final = nn.RNN(hidden_size, final_punct_classes, batch_first=True, dropout=dropout_rate, bidirectional=False)    # Sin puntuación || ? || , || . 
    self.RNN_capitalizacion = nn.RNN(hidden_size, cap_punct_classes, batch_first=True, dropout=dropout_rate, bidirectional=False)      # minúscula || mayúscula inicial || algunas mayúsculas |toda mayúscula 
    self.activation_output = nn.Softmax(dim=1)                  # activación SoftMax para el output
    
  def forward(self, x):
      x = self.embedding(x)
      x, _ = self.RNN(x)

      x_initial_punct, _ = self.RNN_puntuacion_inic(x)
      x_final_punct, _ = self.RNN_puntuacion_final(x)
      x_cap, _ = self.RNN_capitalizacion(x)
      '''
      # Cada bloque lineal produce logits (no probabilidades) para cada clase, en la tarea que le corresponde
      # No agrego softmax o algo del estilo porque luego usamos CrossEntropyLoss, que espera los logits.
      score_puntuacion_inic = self.activation_output(x_initial_punct)
      score_puntuacion_final = self.activation_output(x_final_punct)
      score_capitalizacion = self.activation_output(x_cap)
      '''
      return x_initial_punct, x_final_punct, x_cap #score_puntuacion_inic, score_puntuacion_final, score_capitalizacion
