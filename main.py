import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

'''
IOB_Dataset Class:
    - Reads the data from the CSV file and creates a corpus of token and tag ids
    - If training, creates a vocabulary for the tokens and tags
    - Converts the tokens and tags to their respective ids
    - Pads sequences to the same length

    - __getitem__ returns a token and tag id sequence
    - __len__ returns the number of sequences in the corpus
'''
class IOB_Dataset(Dataset):

    def __init__(self, data, token_vocab=None, tag_vocab=None, training=True):
        self.data = data
        self.training = training

        if training:
            self.token_vocab = {"<PAD>": 0, "<UNK>": 1}
            self.tag_vocab = {"<PAD>": 0, "<UNK>": 1}

            for _, row in self.data.iterrows():
                for token in row["utterances"].split():
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                for tag in row["IOB Slot tags"].split():
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        self.pad_token = self.token_vocab.get("<PAD>", 0)
        self.unk_token = self.token_vocab.get("<UNK>", 0)

        self.corpus_token_ids = []
        self.corpus_tag_ids = []

        for _, row in self.data.iterrows():
            token_ids = [
                self.token_vocab.get(token, self.token_vocab["<UNK>"])
                for token in row["utterances"].split()
            ]

            if "IOB Slot tags" in row and not pd.isna(row["IOB Slot tags"]):
                tag_ids = [
                    self.tag_vocab.get(tag, self.tag_vocab["<UNK>"])
                    for tag in row["IOB Slot tags"].split()
                ]
            else:
                tag_ids = [0] * len(token_ids)

            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_tag_ids.append(torch.tensor(tag_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]

'''
collate_fn:
    - Pads sequences to the same length
    - Returns a padded token and tag id sequence
    - Collate token_ids and tag_ids to make minibatches
'''
def collate_fn(batch):
    # batch [(token_ids, tag_ids), (token_ids, tag_ids), ...]

    # Separate the data into tokens and tags
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]

    # Pad sequences
    padded_sentences = pad_sequence(
        token_ids, batch_first=True, padding_value=train_dataset.token_vocab["<PAD>"]
    )
    padded_tags = pad_sequence(
        tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab["<PAD>"]
    )

    return padded_sentences, padded_tags


EMBEDDING_DIM = 500
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15

data = pd.read_csv("hw2_train.csv")
train_data, val_data = train_test_split(data, test_size=0.1, random_state=seed)

unique_tags = data["IOB Slot tags"].str.split().explode().unique()

exclude_tags = ["<PAD>", "<UNK>"]
unique_tags = np.array([tag for tag in unique_tags if tag not in exclude_tags])

y_train = train_data["IOB Slot tags"].str.split().explode().values
y_train_cleaned = [tag for tag in y_train if tag not in exclude_tags]


# Create separate datasets
train_dataset = IOB_Dataset(data=train_data, training=True)
val_dataset = IOB_Dataset(
    data=val_data,
    training=False,
    token_vocab=train_dataset.token_vocab,
    tag_vocab=train_dataset.tag_vocab,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

'''
SeqTagger Class:
    - Defines the model architecture
    - Embeds the token ids
    - Passes the embedded tokens through an LSTM
    - Passes the LSTM outputs through a linear layer to get the tag scores
'''
class SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        rnn_out, _ = self.lstm(embeddings)  # (batch_size, seql_len, hidden_dim)

        # Check the shape of rnn_out to handle both batch sizes > 1 and 1
        if len(rnn_out.size()) == 3:  # (batch_size, seq_len, hidden_dim * 2)
            batch_size, seq_len, _ = rnn_out.size()
        elif len(rnn_out.size()) == 2:  # (1, seq_len, hidden_dim * 2) when batch_size=1
            batch_size, seq_len = rnn_out.size(0), rnn_out.size(1)
            rnn_out = rnn_out.unsqueeze(0)  # Add a batch dimension to maintain consistency

        rnn_out = rnn_out.contiguous().view(batch_size * seq_len, -1)

        outputs = self.fc(rnn_out)  # (batch_size, seq_len, tagset_size)

        outputs = outputs.view(batch_size, seq_len, -1)

        return outputs
    
tag_counts = train_data["IOB Slot tags"].str.split().explode().value_counts()
all_tags = list(train_dataset.tag_vocab.keys())  # Get all tags including special tokens
weights = torch.ones(len(train_dataset.tag_vocab))  # Initialize weights for all classes

# Calculate weights for non-special tokens
unique_tag_weights = compute_class_weight(
    class_weight='balanced', 
    classes=unique_tags, 
    y=y_train_cleaned
)

model = SeqTagger(
    vocab_size=len(train_dataset.token_vocab),
    tagset_size=len(train_dataset.tag_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
)
loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab["<PAD>"], weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

'''
Train and validate the model
    - Trains the model for a number of epochs
    - Validates the model on the validation set
    - Saves the incorrect predictions from the final epoch to a CSV file
'''
def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn):
    incorrect_predictions = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            optimizer.zero_grad()

            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation Loop
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_tags = []

        # Create reverse mapping for tag IDs to labels
        id2tag = {v: k for k, v in train_dataset.tag_vocab.items()}

        with torch.no_grad():
            for token_ids, tag_ids in val_loader:
                outputs = model(token_ids)
                
                # Calculate loss before reshaping
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
                total_val_loss += loss.item()

                # Get predictions (keep batch dimension)
                predictions = outputs.argmax(dim=-1)  # Shape: [batch_size, seq_len]
                
                # Process each sequence in the batch
                for pred_seq, true_seq in zip(predictions, tag_ids):

                    mask = true_seq != train_dataset.tag_vocab["<PAD>"]

                    # Convert to lists before processing
                    pred_seq = pred_seq.tolist()
                    true_seq = true_seq.tolist()
                    mask = mask.tolist()

                    # Convert to IOB labels and add to lists
                    pred_labels = [id2tag[p] for p, m in zip(pred_seq, mask) if m]
                    pred_labels = [label if label != "<UNK>" else "O" for label in pred_labels]
                    true_labels = [id2tag[t] for t, m in zip(true_seq, mask) if m]

                    if pred_labels != true_labels:
                        incorrect_predictions.append({
                            'Predicted': pred_labels,
                            'True': true_labels
                        })

                    all_predictions.append(pred_labels)
                    all_tags.append(true_labels)

        # Compute train and val loss
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score with sequences of IOB labels
        f1 = f1_score(all_tags, all_predictions, average="weighted")
        print(
            f"{epoch = } | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}"
        )


    incorrect_df = pd.DataFrame(incorrect_predictions)
    incorrect_df.to_csv("incorrect_predictions.csv", index=False)
    print(f"Incorrect predictions saved to file")

    return model



def test_model_on_unseen_data(
    model, test_data, token_vocab, tag_vocab, file_path="test_predictions.csv"
):
    # Create the Dataset for the test data
    test_dataset = IOB_Dataset(
        data=test_data, token_vocab=token_vocab, tag_vocab=tag_vocab, training=False
    )

    # Create reverse mapping for tag_vocab
    id2tag = {v: k for k, v in tag_vocab.items()}


    model.eval()

    all_predictions = []
    all_ids = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            token_ids, tag_ids = test_dataset[idx]
            sample_id = test_data.iloc[idx]["ID"]

            # Add batch dimension for single sequence processing
            token_ids = token_ids.unsqueeze(0)
            
            outputs = model(token_ids)
            # Remove the reshape since model.forward now handles this
            predictions = outputs.argmax(dim=-1)
            
            # Get the predictions for the first (and only) sequence
            predictions = predictions.squeeze(0)
            
            # Get valid predictions (non-padding)
            mask = tag_ids != test_dataset.tag_vocab["<PAD>"]
            
            pred_labels = [id2tag[pred.item()] for pred, m in zip(predictions, mask) if m]
            pred_labels = [label if label != "<UNK>" else "O" for label in pred_labels]

            all_predictions.append(pred_labels)
            all_ids.append(sample_id)

    output = []
    for sample_id, pred_labels in zip(all_ids, all_predictions):
        output.append([sample_id, " ".join(pred_labels)])

    predictions_df = pd.DataFrame(output, columns=["ID", "IOB Slot tags"])

    predictions_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")


test_data = pd.read_csv("hw2_test.csv")
model=train_and_validate(model, train_loader, val_loader, optimizer, loss_fn)
test_model_on_unseen_data(
    model,
    test_data,
    train_dataset.token_vocab,
    train_dataset.tag_vocab,
    file_path="predictions.csv",
)
