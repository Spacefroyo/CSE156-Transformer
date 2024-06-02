# Transformer code for Part 3

import torch
import torch.nn as nn
import torch.nn.functional as F

# The Attention class is a Multi-head attention mechanism
class Attention(nn.Module):

    # A single attention head
    class Head(nn.Module):

        def __init__(self, block_size, n_embd, head_size, device, masked=False):
            super().__init__()

            self.block_size = block_size
            self.n_embd = n_embd
            self.head_size = head_size
            self.masked = masked

            self.K = nn.Linear(n_embd, head_size, bias = False)
            self.Q = nn.Linear(n_embd, head_size, bias = False)
            self.V = nn.Linear(n_embd, head_size, bias = False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, seq):
            keys = self.K(seq)
            queries = self.Q(seq)
            values = self.V(seq)

            # atts is the attention map
            atts = queries @ keys.transpose(-2, 1) / (self.head_size ** 0.5)
            if (self.masked):
                atts = atts.masked_fill(self.tril == 0, float('-inf'))
            atts = F.softmax(atts, dim=-1)

            att_values = atts @ values
            return att_values, atts.detach()


    def __init__(self, block_size, n_head, n_embd, device, masked=False):
        super().__init__()

        self.n_head = n_head
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Attention.Head(block_size, n_embd, head_size, device, masked=masked) for _ in range(n_head)])
        self.resize = nn.Linear(n_head * head_size, n_embd)

    def forward(self, seq):
        attention_maps, heads_out = [], []
        for head in self.heads:
            head_out, attention_map = head(seq)

            attention_maps.append(attention_map)
            heads_out.append(head_out)

        # The output of the Multi-head attention mechanism is the concatenation of the outputs of all its heads
        return self.resize(torch.cat(heads_out, dim=-1)), attention_maps

class Encoder(nn.Module):
    # A block consists of a multi-head attention mechanism and a feedforward, as well as layer norms
    class Block(nn.Module):
        def __init__(self, block_size, n_head, n_embd, device):
            super().__init__()
            self.attention = Attention(block_size, n_head, n_embd, device)
            self.ff = nn.Sequential(
                nn.Linear(n_embd, 4*n_embd),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd)
            )
            self.layer_norm = nn.LayerNorm(n_embd)

        def forward(self, seq):
            # multi-head attention output
            seq = self.layer_norm(seq)
            att_values, attention_map = self.attention(seq)
            att_out = seq + att_values
            
            # feedforward output + residual
            att_out = self.layer_norm(att_out)
            out = att_out + self.ff(att_out)
            
            return out, attention_map
        
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([Encoder.Block(block_size, n_head, n_embd, device) for _ in range(n_layer)])


    def forward(self, seq):
        # Original embedding sequence = token embeddings + positional embeddings
        word_embs = self.token_embeddings(seq)
        pos_embs = self.position_embeddings(torch.arange(self.block_size, device=self.device))

        embs = word_embs + pos_embs

        # Go through the n blocks of attention + feedforward
        attention_maps = []
        for block in self.blocks:
            embs, attention_map = block(embs)
            attention_maps.extend(attention_map)

        return embs, attention_maps




class Decoder(nn.Module):
    # A block consists of a masked multi-head attention mechanism and a feedforward, as well as layer norms
    class Block(nn.Module):
        def __init__(self, block_size, n_head, n_embd, n_hidden, device, cheat=False):
            super().__init__()
            self.attention = Attention(block_size, n_head, n_embd, device, masked = not cheat)
            self.ff = nn.Sequential(
                nn.Linear(n_embd, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_embd)
            )
            self.layer_norm = nn.LayerNorm(n_embd)

        def forward(self, seq):
            # masked multi-head attention output
            seq = self.layer_norm(seq)
            att_values, attention_map = self.attention(seq)
            att_out = seq + att_values
            
            # feedforward output + residual
            att_out = self.layer_norm(att_out)
            out = att_out + self.ff(att_out)

            return out, attention_map
        
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, n_hidden, device, cheat=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([Decoder.Block(block_size, n_head, n_embd, n_hidden, device, cheat) for _ in range(n_layer)])

        self.to_vocab = nn.Linear(n_embd, vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    # Given a sequence, predict a sequence
    def pred(self, seq):
        # Original embedding sequence = token embeddings + positional embeddings
        word_embs = self.token_embeddings(seq)
        pos_embs = self.position_embeddings(torch.arange(self.block_size, device=self.device))

        embs = word_embs + pos_embs

        # Go through the n blocks of attention + feedforward
        attention_maps = []
        for block in self.blocks:
            embs, attention_map = block(embs)
            attention_maps.extend(attention_map)

        # Based on each output token's embedding, generate a (pre-softmax) likelihood for each word in the vocabulary
        preds = self.to_vocab(embs)

        return preds, attention_maps
    
    # Given an input and target sequence, predict an output sequence and return the cross-entropy loss
    def forward(self, seq, target_seq=None):
        preds, attention_maps = self.pred(seq)

        # if no target sequence was given, return the predicted output sequence
        if (target_seq is None):
            return preds, attention_maps
        
        if len(target_seq.shape) == 2:
            target_seq = target_seq.view(-1)
            preds = preds.view(-1, self.vocab_size)
        else:
            target_seq = F.softmax(target_seq.view(-1, self.vocab_size), dim=-1)
            preds = preds.view(-1, self.vocab_size)

        loss = F.cross_entropy(preds, target_seq)

        return loss, attention_maps