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
        d = torch.load(model_path, map_location="cpu")
        # m = d.module
        train_model.load_state_dict(d)
        print("pretrained model loaded")
    else:
        print("training model from scratch")
else:
    print("training model from scratch")

test_data = DTU.DTUDelDataset(cfg, "test")
test_data_loader = DeepDT_dataloader.DataListLoader(test_data, cfg["batch_size"], num_workers=cfg["num_workers"])

for test_data_list in test_data_loader:
    for data in test_data_list:
        data.adj = sparse_mx_to_torch_sparse_tensor(data.adj)

for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
    with torch.cuda.device(d):
        torch.cuda.empty_cache()

for test_data_list in test_data_loader:
    for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()
    for data in test_data_list:
        data.adj = sparse_mx_to_torch_sparse_tensor(data.adj)

    train_model.eval()
    cell_pred, loss1, loss2 = train_model(test_data_list)
    preds = cell_pred.max(dim=1)[1]

    labels_pr = preds.detach().cpu() + 1
    cnt = 0
    for data in test_data_list:
        label_num = data.cell_vertex_idx.shape[0]
        label_begin = cnt
        label_end = cnt + label_num
        cnt += label_num
        data_labels_pr = labels_pr[label_begin:label_end].numpy()
        loss1_pr = torch.mean(loss1[label_begin:label_end]).item()
        loss2_pr = torch.mean(loss2[label_begin:label_end]).item()
        output_dir = os.path.join(cfg["experiment_dir"], data.data_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savetxt(os.path.join(output_dir, "pre_label.txt"), data_labels_pr, fmt='%d')
        print('loss1 %.6f, loss2 %.6f' % (loss1_pr, loss2_pr))
        print("test", data.data_name, "done.")

    for d in [torch.device('cuda:{}'.format(cfg["device_ids"][i])) for i in range(len(cfg["device_ids"]))]:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()




















