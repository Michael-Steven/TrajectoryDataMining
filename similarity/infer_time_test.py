import torch
from models import LSTMSimCLR
from tqdm import tqdm
from collections import OrderedDict
import time
import torchsummary

max_len = 100
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
max_vocab_size = 18866
# max_vocab_size = 23656
device = "cuda:5"
features_type = "encoder"

test_model_path = "./log/porto_tcn/checkpoint_0009_hiddensize_128_batchsize_128_bidirectional_0_nlayers_1_freeze_1.pth.tar"
# test_model_path = "./log/porto_lstm1/checkpoint_0001_hiddensize_128_batchsize_128_bidirectional_0_nlayers_1_freeze_1.pth.tar"
checkpoint = torch.load(test_model_path, map_location=device)
state_dict = checkpoint["state_dict"]

# create new OrderedDict that does not contain 'module'.

model = LSTMSimCLR(max_vocab_size, hidden_size, bidirectional, n_layers)
model.load_state_dict(state_dict)
model.to(device)

trajs = torch.randint(100, 10000, (2, 100), dtype=torch.long)
trajs_len = torch.randint(5, 100, (2, ), dtype=torch.long)
# print(test_input.shape)
# print(seqlen.shape)
# exit()

# torchsummary.summary(model, [(100,), (1,)])
# exit(0)
time1 = time.time()
for i in tqdm(range(1280)):
    trajs, trajs_len = trajs.to(device), trajs_len.to(device)
    features = model.encode_by_encoder(trajs, trajs_len)
time2 = time.time()
print(time2-time1)