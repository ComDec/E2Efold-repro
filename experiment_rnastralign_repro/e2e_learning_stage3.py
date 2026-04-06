import torch.optim as optim
from torch.utils import data

from e2efold.models import ContactNetwork, ContactNetwork_test, ContactNetwork_fc
from e2efold.models import ContactAttention, ContactAttention_simple_fix_PE
from e2efold.models import Lag_PP_NN, RNA_SS_e2e, Lag_PP_zero, Lag_PP_perturb
from e2efold.models import Lag_PP_mixed, Lag_PP_final, ContactAttention_simple
from e2efold.common.utils import *
from e2efold.common.config import process_config
from e2efold.evaluation import all_test_only_e2e

args = get_args()

config_file = args.config

config = process_config(config_file)
print("#####Stage 3#####")
print('Here is the configuration of this run: ')
print(config)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

d = config.u_net_d
BATCH_SIZE = config.BATCH_SIZE
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
pp_loss = config.pp_loss
data_type = config.data_type
model_type = config.model_type
pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position = config.rho_per_position
model_path = '../models_ckpt/supervised_{}_{}_d{}_l3.pt'.format(model_type, data_type, d)
pp_model_path = '../models_ckpt/lag_pp_{}_{}_{}_position_{}.pt'.format(
    pp_type, data_type, pp_loss, rho_per_position)
e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type,
    pp_type, d, data_type, pp_loss, rho_per_position)
epoches_third = config.epoches_third
evaluate_epi = config.evaluate_epi
step_gamma = config.step_gamma
k = config.k

steps_done = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch(0)

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

# Build score network
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

# Build post-processing network (pass L for matrix-mode rho)
if pp_type == 'nn':
    lag_pp_net = Lag_PP_NN(pp_steps, k).to(device)
if 'zero' in pp_type:
    lag_pp_net = Lag_PP_zero(pp_steps, k).to(device)
if 'perturb' in pp_type:
    lag_pp_net = Lag_PP_perturb(pp_steps, k).to(device)
if 'mixed' in pp_type:
    lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position, L=seq_len).to(device)
if 'final' in pp_type:
    lag_pp_net = Lag_PP_final(pp_steps, k, rho_per_position).to(device)

if LOAD_MODEL and os.path.isfile(model_path):
    print('Loading u net model...')
    contact_net.load_state_dict(torch.load(model_path, weights_only=False))
if LOAD_MODEL and os.path.isfile(pp_model_path):
    print('Loading pp model...')
    lag_pp_net.load_state_dict(torch.load(pp_model_path, weights_only=False))

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

if LOAD_MODEL and os.path.isfile(e2e_model_path):
    print('Loading e2e model...')
    rna_ss_e2e.load_state_dict(torch.load(e2e_model_path, weights_only=False))

all_optimizer = optim.Adam(rna_ss_e2e.parameters())

pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')


def val_evaluation():
    contact_net.eval()
    lag_pp_net.eval()
    result_pp = list()
    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in val_generator:
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        state_pad = torch.zeros(contacts.shape).to(device)
        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(PE_batch,
                seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

        final_pred = (a_pred_list[-1].cpu() > 0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp

    pp_exact_p, pp_exact_r, pp_exact_f1 = zip(*result_pp)
    print('Val F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(
        np.average(pp_exact_f1), np.average(pp_exact_p), np.average(pp_exact_r)))


# Training loop
if not args.test:
    all_optimizer.zero_grad()
    for epoch in range(epoches_third):
        rna_ss_e2e.train()
        for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            matrix_reps_batch = torch.unsqueeze(
                torch.Tensor(matrix_reps.float()).to(device), -1)

            contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            state_pad = torch.zeros([matrix_reps_batch.shape[0],
                seq_len, seq_len]).to(device)

            PE_batch = get_pe(seq_lens, seq_len).float().to(device)
            pred_contacts, a_pred_list = rna_ss_e2e(PE_batch,
                seq_embedding_batch, state_pad)

            loss_u = criterion_bce_weighted(pred_contacts * contact_masks, contacts_batch)

            if pp_loss == "l2":
                loss_a = criterion_mse(
                    a_pred_list[-1] * contact_masks, contacts_batch)
                for i in range(pp_steps - 1):
                    loss_a += np.power(step_gamma, pp_steps - 1 - i) * criterion_mse(
                        a_pred_list[i] * contact_masks, contacts_batch)
                mse_coeff = 1.0 / (seq_len * pp_steps)

            if pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1] * contact_masks, contacts_batch)
                for i in range(pp_steps - 1):
                    loss_a += np.power(step_gamma, pp_steps - 1 - i) * f1_loss(
                        a_pred_list[i] * contact_masks, contacts_batch)
                mse_coeff = 1.0 / pp_steps

            loss_a = mse_coeff * loss_a
            loss = loss_u + loss_a

            if steps_done % OUT_STEP == 0:
                print('Stage 3, epoch {}, step: {}, loss_u: {:.4f}, loss_a: {:.4f}, loss: {:.4f}'.format(
                    epoch, steps_done, loss_u, loss_a, loss))

                final_pred = a_pred_list[-1].cpu() > 0.5
                f1 = list(map(lambda i: F1_low_tri(final_pred.cpu()[i],
                    contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
                print('Average training F1 score: ', np.average(f1))

            loss.backward()
            if steps_done % 30 == 0:
                all_optimizer.step()
                all_optimizer.zero_grad()
            steps_done = steps_done + 1

        if epoch % evaluate_epi == 0:
            torch.save(rna_ss_e2e.state_dict(), e2e_model_path)
            val_evaluation()

print('Stage 3 training complete. Model saved to', e2e_model_path)
