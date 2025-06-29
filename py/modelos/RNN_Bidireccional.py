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

class RNN_Bidireccional(nn.Module):
  def __init__(self,
               hidden_size,  
               num_layers, 
               embedding_dim,
               vocab_size,
               cap_class_size=4,
               initial_punct_class_size=2,
               final_punct_class_size=4,
               bert_embedding=False):
    super(RNN_Bidireccional, self).__init__()

    #si tenemos embeddings pre-entrenados, los usamos
    if bert_embedding:
      #self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
      self.embedding = get_multilingual_token_embedding
    else:
      self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)      

    #usamos dos RNNs bidireccionales, según el paper que habíamos visto
    self.rnn1 = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
    ##### self.rnn1 = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
    self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    #usamos tres lineales para predecir la puntuación inicial, la final y la capitalización con la otra
    self.linear_initial_punctuation = nn.Linear(hidden_size, initial_punct_class_size)
    self.linear_final_punctuation = nn.Linear(hidden_size, final_punct_class_size)
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
      x_punt_inicial = self.linear_initial_punctuation(x)
      x_punt_inicial = self.activation_output(x_punt_inicial)

      x_punt_final = self.linear_final_punctuation(x)
      x_punt_final = self.activation_output(x_punt_final)

      x_cap = self.linear_capitalization(x)
      x_cap = self.activation_output(x_cap)

      #return torch.cat((x_punt_inicial,x_punt_final,x_cap), dim=0)
      return x_punt_inicial, x_punt_final, x_cap