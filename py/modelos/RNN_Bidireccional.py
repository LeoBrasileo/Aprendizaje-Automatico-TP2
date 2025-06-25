import torch
import torch.nn as nn

class RNN_Bidireccional(nn.Module):
  def __init__(self, 
               input_size, 
               hidden_size, 
               punct_class_size, 
               cap_class_size, 
               num_layers, 
               embedding_dim,
               vocab_size,
               embeddings=None):
    super(RNN_Bidireccional, self).__init__()

    #si tenemos embeddings pre-entrenados, los usamos
    if embeddings:
      self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    else:
      self.embedding = nn.Embedding(input_size, hidden_size)
      ##### COMENTARIO DE BIANCA: Para mí habría que cambiar a lo que sigue (y también cambiar la rnn1 con lo que comenté abajo), ya que la nn.Embedding define la "look-up table"
      ##### self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    #usamos dos RNNs bidireccionales, según el paper que habíamos visto
    self.rnn1 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    ##### self.rnn1 = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    #usamos dos lineales para predecir la puntuación con una y la capitalización con la otra
    ### COMENTARIO (duda) DE BIANCA: entonces las clases de puntuación son tuplas (punt inicial, punt final) = ( ¿ , ? ), (None, ","),... ? es decir tenemos 8 clases para puntuación en vez de tomarlo como dos tareas separadas? Yo en la clase de la red clásica había hecho como si fuesen 3 tareas.
    self.linear_punctuation = nn.Linear(hidden_size, punct_class_size)
    self.linear_capitalization = nn.Linear(hidden_size, cap_class_size)
    
    #funciones de activación para las capas ocultas (ReLu) y para el ouput una softmax para cada set de predicciones
    self.activation_hidden = nn.ReLU()
    self.activation_output = nn.Softmax(dim=1)

  def forward(self, x):
    #x_in : input_size, x_out : hidden_size
      x = self.embedding(x)

    #x_in : hidden_size, x_out : hidden_size
      x = self.rnn1(x)
      x = self.activation_hidden(x)
      x = self.rnn2(x)
      x = self.activation_hidden(x)

    #x_in : hidden_size, x_out : |clases puntuación| + |clases capitalización|
      x_punt = self.linear_punctuation(x)
      x_punt = self.activation_output(x_punt)

      x_cap = self.linear_capitalization(x)
      x_cap = self.activation_output(x_cap)

      return torch.cat((x_punt,x_cap), dim=0)