import torch
torch.device('cpu')

corpus: str

with open("input.txt", "r") as file:
    corpus = file.read()

vocab: list[str] = sorted(list(set(corpus)))

print(f"vocab length: {len(vocab)}")
print("vocab:")
print("".join(vocab))


### Tokenizer
stoi = {ch: i for (i, ch) in enumerate(vocab)}
itos = {i: ch for (i, ch) in enumerate(vocab)}


def encode(string):
    return [stoi[s] for s in string]


def decode(code):
    return [itos[i] for i in code]


input = "hello world"

print(f"input:{input}")
encoded = encode(input)
print(f"encoded:{encoded}")

decoded = decode(encoded)
print(f"decoded:{decoded}")

data = torch.tensor(encode(corpus), dtype=torch.long)

print(data[:1000])

n = int(len(data) * 0.9)
train_data = data[:n]
test_data = data[n:]

assert (
    train_data.shape[0] + test_data.shape[0] == data.shape[0]
), "the number of elements in training and validation set doesn't add up to full number of elements"

print(f"Number of train elements: {train_data.shape[0]}")
print(f"Number of validation elements: { test_data.shape[0] }")
