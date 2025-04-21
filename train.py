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

block_size = 8
batch_size = 4

torch.manual_seed(1337)
def get_batches(split):
    data = train_data if split == 'train' else test_data
    idxs=[]
    idxs = torch.randint(0, len(train_data)-block_size-1, (batch_size,)).tolist()
    print(idxs)

    xs = torch.stack([data[i:i+block_size] for i in idxs])
    ys = torch.stack([data[i+1:i+block_size+1] for i in idxs])
    print(xs)
    print(ys)

    for i in range(batch_size):
        for j in range(block_size):
            print(f"input {xs[i][0:j]} should give {ys[i][j]}")

get_batches('train')
