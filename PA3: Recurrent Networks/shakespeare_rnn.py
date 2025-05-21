import torch
import torch.nn as nn

# Define RNN Model
# AI prompt: same as the writeup PDF.
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN layer
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)  # (batch_size, sequence_length, embed_size)
        
        # Initialize hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through RNN - only take the last output
        output, hidden = self.rnn(embedded, hidden)
        output = output[:, -1, :]  # Only take the last output
        
        # Pass through final layer
        output = self.fc(output)  # (batch_size, vocab_size)
        
        return output
    

# Define RNN Without Teacher Force Model
# AI prompt: same as the writeup PDF.
class RNNModelWithoutTeacherForce(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN layer
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)
        sequence_length = x.size(1)

        # Embed the input
        embedded = self.embedding(x)  # (batch_size, sequence_length, embed_size)
        
        # Initialize hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        input_timestep = embedded[:, 0, :].unsqueeze(1) 

        for _ in range(sequence_length):
            output, (h0, c0) = self.lstm(input_timestep, (h0, c0))  
            output_logits = self.fc(output.squeeze(1))
            predicted_idx = output_logits.argmax(dim=1)
            input_timestep = self.embedding(predicted_idx).unsqueeze(1)

        return output_logits