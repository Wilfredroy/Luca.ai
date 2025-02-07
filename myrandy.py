import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_seq_len, tgt_seq_len = src.size(1), tgt.size(1)
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_len, :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_len, :]
        output = self.transformer(src, tgt)
        return self.fc_out(output)

# Hyperparameters
vocab_size = 10000  # Size of the vocabulary
embed_dim = 512     # Embedding dimension
num_heads = 8       # Number of attention heads
num_layers = 6      # Number of transformer layers
max_seq_len = 128   # Maximum sequence length

# Initialize the model
model = SimpleTransformer(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)

# Example tokenized dataset
data = [
    [1, 2, 3, 4, 5],  # Sentence 1
    [6, 7, 8, 9, 10], # Sentence 2
    # Add more sentences...
]

# Convert to PyTorch tensors
src_data = torch.tensor(data[:-1], dtype=torch.long)  # Source (input)
tgt_data = torch.tensor(data[1:], dtype=torch.long)   # Target (output)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(src_data, tgt_data[:-1])  # Predict next token
    loss = criterion(output.view(-1, vocab_size), tgt_data[1:].view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")