# coding: UTF-8
import torch
import time
import numpy as np
import argparse
from importlib import import_module
from util import build_dataset, build_iterator, build_dataset_bert
from util import get_time_dif
from train_eval import train, train_bert


parser = argparse.ArgumentParser(description="text classification...  dnn")
parser.add_argument("--model",
                    type=str,
                    required=True,
                    help=" TextCNN TextRNN FastText TextRCNN Transformer bert")
parser.add_argument('--embedding',
                    default='pre_trained',
                    type=str,
                    help='random or pre_trained')
parser.add_argument('--word',
                    default=False,
                    type=bool,
                    help='True for word, False for char')
args = parser.parse_args()

if __name__ == "__main__":
    data_set = "data"
    embedding = "embedding_SougouNews.npz"
    model_name = args.model
    model = import_module("models." + model_name)  # 导入model
    config = model.Config(data_set, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #  加载数据
    start_time = time.time()
    print("start loading data...\n")

    vocab, train_data, dev_data, test_data = build_dataset(
        config, args.word)
    config.n_vocab = len(vocab)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("\nload data time usage : ", time_dif)
    # 开始训练
    start_time = time.time()
    model = model.Model(config).to(config.device)
    print("------\nmodel : {0}\n".format(model_name))
    print(model.parameters)
    print("------\ntraining...\n")

    train(config, model, train_iter, dev_iter, test_iter)

    time_dif = get_time_dif(start_time)
    print("\ntrain time usage: ", time_dif)
