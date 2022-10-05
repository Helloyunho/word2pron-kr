# Pretty much based on official PyTorch seq2seq tutorial

from xlrd import open_workbook
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import math
from pathlib import Path

lr = 0.001
n_hidden = 256
n_iters = 1000000
print_term = 500
enc_snapshot_path = Path("enc_snapshot.pt")
dec_snapshot_path = Path("dec_snapshot.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_pronounce = {}


def remove_other_char(word):
    return re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", word)


for xls in glob.glob("data/*.xls"):
    wb = open_workbook(xls)
    sheet = wb.sheet_by_index(0)
    for r in range(1, sheet.nrows):
        word = remove_other_char(sheet.cell(r, 0).value)
        if word in word_pronounce or word == "":
            continue
        pronounce = sheet.cell(r, 8).value
        if pronounce == "":
            pronounce = word
        else:
            p = re.search(r"^\[([^\/\]]+)(?:\/[^\/\]]+)*\]$", pronounce, re.MULTILINE)
            if p:
                pronounce = remove_other_char(p.group(1))
        word_pronounce[word] = pronounce

randomized_set = random.choices(list(word_pronounce.items()), k=len(word_pronounce))

trainset = dict(randomized_set[len(word_pronounce) // 8 :])
testset = dict(randomized_set[: len(word_pronounce) // 8])

start_seq = "\x02"
end_seq = "\x03"
all_letters = "\x02\x03 "
for i in range(
    int.from_bytes("가".encode("utf-16")[2:4], "little"),
    int.from_bytes("힣".encode("utf-16")[2:4], "little") + 1,
):
    all_letters += chr(i)

for i in range(
    int.from_bytes("ㄱ".encode("utf-16")[2:4], "little"),
    int.from_bytes("ㅎ".encode("utf-16")[2:4], "little") + 1,
):
    all_letters += chr(i)

for i in range(
    int.from_bytes("ㅏ".encode("utf-16")[2:4], "little"),
    int.from_bytes("ㅣ".encode("utf-16")[2:4], "little") + 1,
):
    all_letters += chr(i)

n_letters = len(all_letters)


def wordToTensor(word):
    indexes = [all_letters.find(c) for c in word]
    indexes.append(all_letters.find(end_seq))
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


class Word2PronKREncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word2PronKREncoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.encoder(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Word2PronKRDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Word2PronKRDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.decoder(embedded, hidden)
        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(
    encoder: Word2PronKREncoder,
    decoder: Word2PronKRDecoder,
    enc_optimizer,
    dec_optimizer,
    criterion,
    iter_index: int,
    teacher_forcing_ratio=0.5,
):
    start = 64 * iter_index
    end = 64 * (iter_index + 1)
    if start >= len(trainset):
        start -= len(trainset) * (start // len(trainset))
        end -= len(trainset) * (end // len(trainset))
    elif end > len(trainset):
        end = len(trainset)
    train_batch = list(trainset.items())[start:end]
    # print(len(train_batch))

    enc_hidden = encoder.initHidden()

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    for word, pronounce in train_batch:
        input_tensor = wordToTensor(word)
        input_length = input_tensor.size(0)
        target_tensor = wordToTensor(pronounce)
        target_length = target_tensor.size(0)

        loss = 0

        for i in range(input_length):
            _, enc_hidden = encoder(input_tensor[i], enc_hidden)

        dec_input = torch.tensor([[all_letters.find(start_seq)]], device=device)
        dec_hidden = enc_hidden

        teacher_forcing = random.random() < teacher_forcing_ratio
        if teacher_forcing:
            for i in range(target_length):
                dec_output, dec_hidden = decoder(dec_input, dec_hidden)
                loss += criterion(dec_output, target_tensor[i])
                dec_input = target_tensor[i]
        else:
            for i in range(target_length):
                dec_output, dec_hidden = decoder(dec_input, dec_hidden)
                loss += criterion(dec_output, target_tensor[i])
                _, topi = dec_output.topk(1)
                dec_input = topi.squeeze().detach()
                if dec_input.item() == all_letters.find(end_seq):
                    break

        loss.backward()

        enc_optimizer.step()
        dec_optimizer.step()

        return loss.item() / target_length
    return 100


def evaluate(encoder, decoder, iter_index: int):
    start = 64 * iter_index
    end = 64 * (iter_index + 1)
    if start >= len(testset):
        start -= len(testset) * (start // len(testset))
        end -= len(testset) * (end // len(testset))
    elif end > len(testset):
        end = len(testset)
    test_batch = list(testset.items())[start:end]
    # print(len(test_batch))

    with torch.no_grad():
        correct = 0
        for word, pronounce in test_batch:
            input_tensor = wordToTensor(word)
            input_length = input_tensor.size(0)

            enc_hidden = encoder.initHidden()

            for i in range(input_length):
                _, enc_hidden = encoder(input_tensor[i], enc_hidden)

            dec_input = torch.tensor([[all_letters.find(start_seq)]], device=device)
            dec_hidden = enc_hidden

            decoded_chars = []

            for i in range(9999999999):  # 9999999999 is just a random number
                dec_output, dec_hidden = decoder(dec_input, dec_hidden)
                _, topi = dec_output.data.topk(1)
                if topi.item() == all_letters.find(end_seq):
                    break
                else:
                    decoded_chars.append(all_letters[topi.item()])
                dec_input = topi.squeeze().detach()

            if pronounce == "".join(decoded_chars):
                correct += 1

        size = len(test_batch)
        if size != 0:
            return correct / size
        else:
            return 0


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


if __name__ == "__main__":
    start = time.time()
    encoder = Word2PronKREncoder(n_letters, n_hidden).to(device)
    decoder = Word2PronKRDecoder(n_letters, n_hidden).to(device)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if enc_snapshot_path.exists():
        encoder.load_state_dict(torch.load(enc_snapshot_path))
    if dec_snapshot_path.exists():
        decoder.load_state_dict(torch.load(dec_snapshot_path))

    val_acc = evaluate(encoder, decoder, 0)
    print("initial accuracy: %.4f" % (val_acc * 100))

    best = None
    for iter in range(1, n_iters + 1):
        val_loss = train(
            encoder, decoder, enc_optimizer, dec_optimizer, criterion, iter - 1
        )
        val_acc = evaluate(encoder, decoder, iter - 1)

        # Print iter number, loss, name and guess
        if iter % print_term == 0:
            print(
                "%s (%d %d%%) %.4f%% %.4f%%"
                % (
                    timeSince(start),
                    iter,
                    iter / n_iters * 100,
                    val_acc * 100,
                    val_loss * 100,
                )
            )

        if best is None or val_loss < best:
            best = val_loss
            torch.save(encoder.state_dict(), enc_snapshot_path)
            torch.save(decoder.state_dict(), dec_snapshot_path)
