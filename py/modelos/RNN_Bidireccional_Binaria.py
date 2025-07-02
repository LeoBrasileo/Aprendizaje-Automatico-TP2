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

        if bert_embedding:
            self.embedding = get_multilingual_token_embedding
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)      

        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
                   
        self.foward_recurrent = nn.RNN(hidden_size*2, initial_punct_class_size + final_punt_class_size + cap_class_size, batch_first=True, bidirectional=False)

        self.cap_class_size = cap_class_size
        self.final_punt_class_size = final_punt_class_size
        self.initial_punct_class_size = initial_punt_class_size


        self.activation_hidden = nn.ReLU()
        self.activation_output = nn.Sigmoid(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.activation_hidden(x)

        x, _ = self.foward_current(x)
      
        x = self.activation_output(x)
        out_init = x[:self.initial_punt_class_size]
        out_final = x[self.initial_punt_class_size : self.final_punt_class_size]
        out_cap = x[self.final_punt_class_size : self.cap_class_size]
        
        return out_init, out_final, out_cap
