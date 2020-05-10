# coding: UTF-8
import torch
import time
import numpy as np
import argparse
from importlib import import_module
from util import build_dataset, build_iterator, build_dataset_bert
from util import get_time_dif
from train_eval import train, train_bert


parser = argparse.ArgumentParser(description="text classification (Bert)")
parser.add_argument("--model",
                    type=str,
                    required=True,
                    help=" Bert / Bert_CNN / Bert_RNN / Bert_RCNN / Bert_HAN")
args = parser.parse_args()

if __name__ == "__main__":
    data_set = "data"
    model_name = args.model
    x = import_module("models." + model_name)  # 导入model
    config = x.Config(data_set)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #  加载数据
    start_time = time.time()
    print("start loading data...\n")
    train_data, dev_data, test_data = build_dataset_bert(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("\nload data time usage : ", time_dif)
    # 开始训练
    start_time = time.time()
    model = x.Model(config).to(config.device)

    print("------\nmodel : {0}\n".format(model_name))
    print(model.parameters)
    print("------\ntraining...\n")

    train(config, model, train_iter, dev_iter, test_iter, is_bert=True)

    time_dif = get_time_dif(start_time)
    print("\ntrain time usage: ", time_dif)
