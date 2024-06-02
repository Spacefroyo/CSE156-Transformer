import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

import transformer as transformer
import transformer_part3
from utilities import Utilities

from matplotlib import pyplot as plt

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def valid_test_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= 2 * eval_iters: break

    valid_losses, test_losses = torch.tensor(losses[eval_iters:]), torch.tensor(losses[:eval_iters])
    mean_valid_loss, mean_test_loss = valid_losses.mean(), test_losses.mean()
    valid_perplexity, test_perplexity = torch.exp(mean_valid_loss).item(), torch.exp(mean_test_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return valid_perplexity, test_perplexity

def main(part_to_run):
    global max_iters

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
  


    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    politicians = ["obama", "hbush", "wbush"]
    test_LM_loaders = {}
    for politician in politicians:
        inputfile = f"speechesdataset/test_LM_{politician}.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=True)
        test_LM_loaders[politician] = test_LM_loader

    part_string = "all parts" if part_to_run == 0 else "part " + str(part_to_run)
    print(f"Currently running {part_string}")








    # Part 1: Classifier
    if part_to_run == 0 or part_to_run == 1:
        encoder = transformer.Encoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, device)

        class MeanEmbdAcrossSeq(nn.Module):
            def forward(self, x):
                seq, _ = x
                return torch.mean(seq, dim=1)
        
        classifier = nn.Sequential(
            encoder,
            MeanEmbdAcrossSeq(),
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        ).to(device)
        print("Classifier # of Parameters:", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
        # for the classification  task, you will train for a fixed number of epochs like this:
        for epoch in range(epochs_CLS):
            epoch_loss, epoch_samples = 0, 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                preds = classifier(xb)

                loss = criterion(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.sum().item()
                epoch_samples += 1

            acc = compute_classifier_accuracy(classifier, test_CLS_loader)
            print(f"Classifier Epoch {epoch+1}:\n\tTest Accuracy: {acc}\n\tTrain Loss: {epoch_loss/epoch_samples}")

        util = Utilities(tokenizer, encoder, device)
        util.sanity_check("How are you doing today", block_size)










    # Part 2: Decoder
    if part_to_run == 0 or part_to_run == 2:
        decoder = transformer.Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, n_hidden, device).to(device)

        print("Decoder # of Parameters:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

        epoch_loss, epoch_samples = 0, 0
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)

            loss, _ = decoder(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.sum().item()
            epoch_samples += 1

            if (i+1) % eval_interval == 0:
                perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
                print(f"Step {i+1} Train Decoder Perplexity:", perplexity)

        for politician in politicians:
            perplexity = compute_perplexity(decoder, test_LM_loaders[politician], eval_iters=eval_iters)
            print(f"Test Decoder Perplexity for {politician}:", perplexity)

        util = Utilities(tokenizer, decoder, device)
        util.sanity_check("The issue of social welfare has been tremendously important to the American public. To address this issue, we must implement new programs designed for the benefit of", block_size)











    # Part 3: Exploration
    if part_to_run == 0 or part_to_run == 3:
        max_iters = 5000
        min_improvement = 0.

        decoder = transformer.Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, n_hidden, device).to(device)

        print("Decoder # of Parameters:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

        decoder_history = []
        epoch_loss, epoch_samples = 0, 0
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        i = 0
        while (i < max_iters):
            for xb, yb in train_LM_loader:
                if (i >= max_iters):
                    break
                xb, yb = xb.to(device), yb.to(device)

                loss, _ = decoder(xb, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.sum().item()
                epoch_samples += 1

                if (i+1) % eval_interval == 0:
                    perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)

                    mean_valid_perplexity = 0
                    for politician in politicians:
                        valid_perplexity, _ = valid_test_perplexity(decoder, test_LM_loaders[politician], eval_iters=eval_iters)
                        mean_valid_perplexity += valid_perplexity
                    mean_valid_perplexity /= len(politicians)
                    print(f"Decoder Step {i+1} Train Perplexity:", perplexity, "\t Valid Perplexity:", mean_valid_perplexity)

                    decoder_history.append(mean_valid_perplexity)

                    if len(decoder_history) > 1 and mean_valid_perplexity + min_improvement > decoder_history[-2]:
                        i = max_iters - 1

                i += 1

        for politician in politicians:
            perplexity = compute_perplexity(decoder, test_LM_loaders[politician], eval_iters=eval_iters)
            print(f"Decoder Test Perplexity for {politician}:", perplexity)

        util = Utilities(tokenizer, decoder, device)
        util.sanity_check("The issue of social welfare has been tremendously important to the American public. To address this issue, we must implement new programs designed for the benefit of", block_size)
        
        cheater = transformer_part3.Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, n_hidden, device, cheat=True).to(device)
        matcher = transformer_part3.Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, n_hidden, device, cheat=False).to(device)

        print("Cheater # of Parameters:", sum(p.numel() for p in cheater.parameters() if p.requires_grad))

        cheater_history = []
        optimizer = torch.optim.Adam(cheater.parameters(), lr=learning_rate)
        i = 0
        while (i < max_iters):
            for xb, yb in train_LM_loader:
                if (i >= max_iters):
                    break
                xb, yb = xb.to(device), yb.to(device)

                loss, _ = cheater(xb, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % eval_interval == 0:
                    perplexity = compute_perplexity(cheater, train_LM_loader, eval_iters=eval_iters)

                    mean_valid_perplexity = 0
                    for politician in politicians:
                        valid_perplexity, _ = valid_test_perplexity(cheater, test_LM_loaders[politician], eval_iters=eval_iters)
                        mean_valid_perplexity += valid_perplexity
                    mean_valid_perplexity /= len(politicians)
                    print(f"Cheater Step {i+1} Train Perplexity:", perplexity, "\t Valid Perplexity:", mean_valid_perplexity)

                    cheater_history.append(mean_valid_perplexity)

                    if len(cheater_history) > 1 and mean_valid_perplexity + min_improvement > cheater_history[-2]:
                        i = max_iters - 1
                
                i += 1

        for politician in politicians:
            perplexity = compute_perplexity(cheater, test_LM_loaders[politician], eval_iters=eval_iters)
            print(f"Cheater Test Perplexity for {politician}:", perplexity)

        util = Utilities(tokenizer, cheater, device)
        util.sanity_check("The issue of social welfare has been tremendously important to the American public. To address this issue, we must implement new programs designed for the benefit of", block_size)

        print("Matcher # of Parameters:", sum(p.numel() for p in cheater.parameters() if p.requires_grad))

        matcher_history = []
        optimizer = torch.optim.Adam(matcher.parameters(), lr=learning_rate)
        i = 0
        while (i < max_iters):
            for xb, yb in train_LM_loader:
                if (i >= max_iters):
                    break
                xb, yb = xb.to(device), yb.to(device)

                cyb, _ = cheater(xb)

                loss, _ = matcher(xb, cyb.detach())

                true_loss, _ = matcher(xb, yb)

                optimizer.zero_grad()
                (loss + true_loss).backward()
                optimizer.step()

                if (i+1) % eval_interval == 0:
                    perplexity = compute_perplexity(matcher, train_LM_loader, eval_iters=eval_iters)

                    mean_valid_perplexity = 0
                    for politician in politicians:
                        valid_perplexity, _ = valid_test_perplexity(matcher, test_LM_loaders[politician], eval_iters=eval_iters)
                        mean_valid_perplexity += valid_perplexity
                    mean_valid_perplexity /= len(politicians)
                    print(f"Matcher Matching Step {i+1} Train Perplexity:", perplexity, "\t Valid Perplexity:", mean_valid_perplexity)

                    matcher_history.append(mean_valid_perplexity)

                    if len(matcher_history) > 1 and mean_valid_perplexity + min_improvement > matcher_history[-2]:
                        i = max_iters - 1

                i += 1

        i = 0
        while (i < max_iters):
            for xb, yb in train_LM_loader:
                if (i >= max_iters):
                    break
                xb, yb = xb.to(device), yb.to(device)

                true_loss, _ = matcher(xb, yb)

                optimizer.zero_grad()
                true_loss.backward()
                optimizer.step()

                if (i+1) % eval_interval == 0:
                    perplexity = compute_perplexity(matcher, train_LM_loader, eval_iters=eval_iters)

                    mean_valid_perplexity = 0
                    for politician in politicians:
                        valid_perplexity, _ = valid_test_perplexity(matcher, test_LM_loaders[politician], eval_iters=eval_iters)
                        mean_valid_perplexity += valid_perplexity
                    mean_valid_perplexity /= len(politicians)
                    print(f"Matcher Self Step {i+1} Train Perplexity:", perplexity, "\t Valid Perplexity:", mean_valid_perplexity)

                    matcher_history.append(mean_valid_perplexity)

                    if len(matcher_history) > 1 and mean_valid_perplexity + min_improvement > matcher_history[-2]:
                        i = max_iters - 1

                i += 1

        for politician in politicians:
            perplexity = compute_perplexity(matcher, test_LM_loaders[politician], eval_iters=eval_iters)
            print(f"Matcher Test Perplexity for {politician}:", perplexity)

        util = Utilities(tokenizer, matcher, device)
        util.sanity_check("The issue of social welfare has been tremendously important to the American public. To address this issue, we must implement new programs designed for the benefit of", block_size)

        plt.plot(decoder_history, label="Normal decoder")
        plt.plot(cheater_history, label="Cheater")
        plt.plot(matcher_history, label="Matcher")
        plt.legend()
        plt.xlabel("Training steps (100s)")
        plt.ylabel("Validation perplexity")
        plt.title("Validation perplexity vs training")

        plt.savefig('normal_vs_cheatermatcher.png')
        
        


if __name__ == "__main__":
    # 0 means run all parts
    part_to_run = int(sys.argv[1])

    main(part_to_run)
