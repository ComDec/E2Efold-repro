import _pickle as pickle

import torch.optim as optim
from torch.utils import data

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
from e2efold.postprocess import postprocess

args = get_args()

config_file = args.config

config = process_config(config_file)
print("#####Stage 1#####")
print('Here is the configuration of this run: ')
print(config)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

d = config.u_net_d
BATCH_SIZE = config.batch_size_stage_1
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
data_type = config.data_type
model_type = config.model_type
model_path = '../models_ckpt/supervised_{}_{}_d{}_l3.pt'.format(model_type, data_type, d)
epoches_first = config.epoches_first
evaluate_epi = config.evaluate_epi_stage_1

steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch()

from e2efold.data_generator import RNASSDataGenerator, Dataset
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data',
    'seq ss_label length name pairs')

# No upsampling - mxfold2 IDs don't have '/' separators needed by upsampling_data()
train_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'train', False)
val_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'val')

seq_len = train_data.data_y.shape[-2]
print('Max seq length ', seq_len)

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

if model_type == 'test_lc':
    contact_net = ContactNetwork_test(d=d, L=seq_len).to(device)
if model_type == 'att6':
    contact_net = ContactAttention(d=d, L=seq_len).to(device)
if model_type == 'att_simple':
    contact_net = ContactAttention_simple(d=d, L=seq_len).to(device)
if model_type == 'att_simple_fix':
    contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len,
        device=device).to(device)
if model_type == 'fc':
    contact_net = ContactNetwork_fc(d=d, L=seq_len).to(device)
if model_type == 'conv2d_fc':
    contact_net = ContactNetwork(d=d, L=seq_len).to(device)

if LOAD_MODEL and os.path.isfile(model_path):
    print('Loading u net model...')
    contact_net.load_state_dict(torch.load(model_path, weights_only=False))

u_optimizer = optim.Adam(contact_net.parameters())

pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight)


def model_eval():
    contact_net.eval()
    contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(val_generator))
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()).to(device), -1)

    state_pad = torch.zeros([matrix_reps_batch.shape[0],
        seq_len, seq_len]).to(device)
    PE_batch = get_pe(seq_lens, seq_len).float().to(device)

    with torch.no_grad():
        pred_contacts = contact_net(PE_batch,
            seq_embedding_batch, state_pad)

    u_no_train = postprocess(pred_contacts,
        seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
    map_no_train = (u_no_train > 0.5).float()
    f1_no_train_tmp = list(map(lambda i: F1_low_tri(map_no_train.cpu()[i],
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
    print('Average val F1 score with pure post-processing: ', np.average(f1_no_train_tmp))


# Training loop
for epoch in range(epoches_first):
    contact_net.train()

    for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).to(device), -1)

        state_pad = torch.zeros([matrix_reps_batch.shape[0],
            seq_len, seq_len]).to(device)

        PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
        pred_contacts = contact_net(PE_batch,
            seq_embedding_batch, state_pad)

        loss_u = criterion_bce_weighted(pred_contacts * contact_masks, contacts_batch)

        if steps_done % OUT_STEP == 0:
            print('Stage 1, epoch: {}, step: {}, loss: {}'.format(
                epoch, steps_done, loss_u))

        u_optimizer.zero_grad()
        loss_u.backward()
        u_optimizer.step()
        steps_done = steps_done + 1

    if epoch % evaluate_epi == 0:
        model_eval()
        torch.save(contact_net.state_dict(), model_path)

print('Stage 1 training complete. Model saved to', model_path)
