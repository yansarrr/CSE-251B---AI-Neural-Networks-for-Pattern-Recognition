import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size // num_layers  # Adjust to keep total parameters similar
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, vocab_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)  # (batch_size, sequence_length, embed_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        output, _ = self.lstm(embedded, (h0, c0))
        output = output[:, -1, :]  # Take last time step output
        
        # Pass through final layer
        output = self.fc(output)  # (batch_size, vocab_size)
        
        return output
    

# Define the LSTM model without teacher forcing
class LSTMModelWithoutTeacherForce(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModelWithoutTeacherForce, self).__init__()
        
        self.hidden_size = hidden_size // num_layers
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, vocab_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        # Embed the input
        embedded = self.embedding(x)  # (batch_size, sequence_length, embed_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        input_timestep = embedded[:, 0, :].unsqueeze(1)

        for _ in range(sequence_length):
            output, (h0, c0) = self.lstm(input_timestep, (h0, c0))  
            output_logits = self.fc(output.squeeze(1))
            predicted_idx = output_logits.argmax(dim=1)
            input_timestep = self.embedding(predicted_idx).unsqueeze(1)

        return output_logits
