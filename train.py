text: str

with open("input.txt", "r") as file:
    text = file.read()

vocab: list[str] = sorted(list(set(text)))

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


