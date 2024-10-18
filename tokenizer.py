import urllib.request
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

file_path = "data.txt"

urllib.request.urlretrieve(url, file_path)

with open("data.txt", 'r', encoding = "utf-8") as f:
    raw_text = f.read()

print(f"Total Characters: {len(raw_text)}")
preprocessed = re.findall(r"[\w']+|[.,:;!?()\"--]", raw_text)

print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab_size = len(all_words)
print(f"Vocab Size: {vocab_size}")

vocab = {token:id for id, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    if i >= 10:
        break
    else:
        print(item)



class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.findall(r"[\w']+|[.,:;!?()\"--]", text)
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text


tokenizer = Tokenizer(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))


bpe_tokenizer = tiktoken.get_encoding("gpt2")

with open("data.txt", "r", encoding = "utf-8") as target:
    raw_text = target.read()

enc_text = bpe_tokenizer.encode(raw_text)

print(len(enc_text))

enc_sample = enc_text[50:]
context = 4
x = enc_sample[:context]
y = enc_sample[1:context+1]

print(x)
print(y)


for i in range(1, context+1):
    context_input = enc_sample[:i]
    desired = enc_sample[i]
    print(context_input, "---->", desired)





class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids)-max_len, stride):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size = 4, max_len = 256, stride = 128, 
                         shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_len, stride)

    dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            drop_last = drop_last,
            num_workers = num_workers
            )

    return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_len=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)

inputs, target = first_batch

print(inputs)
print(target)

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = 4

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
