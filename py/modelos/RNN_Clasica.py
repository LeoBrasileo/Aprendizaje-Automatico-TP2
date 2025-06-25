import torch
import torch.nn as nn

class RNN_Clasica(nn.Module):
  def __init__(self, 
               hidden_size,
               embedding_dim, 
               vocab_size, 
               embeddings=None,
               dropout_rate=0.5
               ):
    
    super(RNN_Clasica, self).__init__()
    
    # Si tenemos embeddings pre-entrenados, los usamos
    if embeddings:
      self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    else:
      self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    # Dos capas LSTM 
    self.LSTM_1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
    self.LSTM_2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    
    # Dropout
    self.dropout = nn.Dropout(dropout_rate)
    
    # Capas de output para las tres tareas que queremos 
    self.linear_puntuacion_inic = nn.Linear(hidden_size, 2)     # Sin puntuación || ¿
    self.linear_puntuacion_final = nn.Linear(hidden_size, 4)    # Sin puntuación || ? || , || . 
    self.linear_capitalizacion = nn.Linear(hidden_size, 4)      # minúscula || mayúscula inicial || algunas mayúsculas || toda mayúscula 

  def forward(self, x):
      x = self.embedding(x)

      x, _ = self.LSTM_1(x)
      x = self.dropout(x)
      x, _ = self.LSTM_2(x)
      x = self.dropout(x)

      # Cada bloque lineal produce logits (no probabilidades) para cada clase, en la tarea que le corresponde
      # No agrego softmax o algo del estilo porque luego usamos CrossEntropyLoss, que espera los logits.
      score_puntuacion_inic = self.linear_puntuacion_inic(x)  
      score_puntuacion_final = self.linear_puntuacion_final(x)   
      score_capitalizacion = self.linear_capitalizacion(x)           

      return score_puntuacion_inic, score_puntuacion_final, score_capitalizacion