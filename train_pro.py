import argparse
import DeepDT_util
import DTU
from DeepDT_data import *
from DeepDT_Parallel import *
import R_GCN_model
import DeepDT_dataloader
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file')

args = parser.parse_args()
cfg = DeepDT_util.load_config(args.config)
cfg = DeepDT_util.augment_config(cfg)
cfg = DeepDT_util.check_config(cfg)

print(cfg)

if not os.path.exists(cfg["experiment_dir"]):
    raise RuntimeError("experiment_dir does not exist.")

if cfg['use_normal']:
    geo_in = 7
else:
    geo_in = 1

train_model = R_GCN_model.R_GCN(geo_in)
model_path = cfg["model_path"]

if cfg["cuda"]:
    train_model = DeepDTParallel(train_model, device_ids=cfg["device_ids"])
    device = torch.device("cuda:{}".format(cfg["device_ids"][0]))
    train_model = train_model.to(device)

if cfg["pretrained"]:
    if os.path.exists(model_path):
        train_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("pretrained model loaded")
    else:
        print("training model from scratch")
else:
    print("training model from scratch")

optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)
train_data = DTU.DTUDelDataset(cfg, "train")
train_data_loader = DeepDT_dataloader.DataListLoader(train_data, cfg["batch_size"], num_workers=cfg["num_workers"])


step_cnt = 0
best_accu = 0.0
best_loss = 0.0
output_dir = os.path.split(model_path)[0]
accu_path = os.path.join(output_dir, "accuracy.txt")
if os.path.exists(accu_path):
    with open(os.path.join(output_dir, "accuracy.txt"), 'r') as f:
        best_accu = float(f.read())
    print("accuracy loaded", best_accu)

init_epoch = 0
epoch_path = os.path.join(output_dir, "epoch.txt")
if os.path.exists(epoch_path):
    with open(os.path.join(output_dir, "epoch.txt"), 'r') as f:
        init_epoch = int(f.read())
    print("init_epoch loaded", init_epoch)
print("init_epoch", init_epoch)
data_lists = []
for data_list in train_data_loader:
    data_lists.append(data_list)
    for data in data_list:
        print(data.label_weights)

for data_list in data_lists:
    for data in data_list:
        data.adj = sparse_mx_to_torch_sparse_tensor(data.adj)



weight2 = 1.0 * cfg["weight_ratio"] / (cfg["weight_ratio"] + 1)
weight1 = 1.0 - weight2
print("loss1 weight", weight1, 'loss2 weight', weight2)

tmp_cnt = 0
tmp_loss1 = 0.0
tmp_label0 = 0
tmp_label1 = 0
tmp_loss2 = 0.0
tmp_loss = 0.0

for epoch in range(init_epoch, cfg['epochs']):
    for data_list in data_lists:
        for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        train_model.train()
        optimizer.zero_grad()

        cell_pred, loss1, loss2 = train_model(data_list)
        loss11 = torch.mean(loss1)
        loss21 = torch.mean(loss2)
        loss = weight1 * loss11 + weight2 * loss21
        loss.backward()
        optimizer.step()

        step_cnt += 1
        cell_pred_label = cell_pred.max(dim=1)[1]
        label0_num = torch.sum(cell_pred_label == 0).item()
        label1_num = torch.sum(cell_pred_label == 1).item()
        outstr = "Train epoch %d, step %d, loss %.6f, loss1 %.6f, loss2 %.6f" \
                 % (epoch, step_cnt, loss.detach().item(), loss11.detach().item(), loss21.detach().item())
        
        print(outstr)

        for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()

        tmp_loss1 += loss11.detach().item()
        tmp_loss2 += loss21.detach().item()
        tmp_loss += loss.detach().item()
        tmp_cnt += 1
        tmp_label0 += label0_num
        tmp_label1 += label1_num

    tmp_loss1 /= tmp_cnt
    tmp_loss2 /= tmp_cnt
    tmp_loss /= tmp_cnt

    outstr = "Test epoch %d, step %d, loss %.6f, loss1 %.6f, loss2 %.6f" \
         % (epoch, step_cnt, tmp_loss, tmp_loss1, tmp_loss2)
    print(outstr)

    if epoch % 50 == 0:
        extra_output_dir = os.path.split(model_path)[0] + '_epoch_' + str(epoch)
        if not os.path.exists(extra_output_dir):
            os.mkdir(extra_output_dir)
        extra_model_path = os.path.join(extra_output_dir, os.path.split(model_path)[1])
        print("Saving extra model", cfg["weight_ratio"])
        torch.save(train_model.state_dict(), extra_model_path)
        loss_array = np.asarray([tmp_loss, tmp_loss1, tmp_loss2])
        np.savetxt(os.path.join(extra_output_dir, 'loss.txt'), loss_array, fmt='%f')


        with open(os.path.join(extra_output_dir, "epoch.txt"), 'w') as f:
            f.write(str(epoch))

    output_dir = os.path.split(model_path)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("Saving model", cfg["weight_ratio"])
    torch.save(train_model.state_dict(), model_path)
    loss_array = np.asarray([tmp_loss, tmp_loss1, tmp_loss2])
    np.savetxt(os.path.join(extra_output_dir, 'loss.txt'), loss_array, fmt='%f')


    with open(os.path.join(output_dir, "epoch.txt"), 'w') as f:
        f.write(str(epoch))

    tmp_cnt = 0
    tmp_loss1 = 0.0
    tmp_label0 = 0
    tmp_label1 = 0
    tmp_loss2 = 0.0
    tmp_loss = 0.0




















