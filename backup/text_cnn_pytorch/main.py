import os
import sys
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from datetime import timedelta
import time
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from model import TextCnnConfig, TextCnn


data_dir = 'data/cnews'
train_dir = os.path.join(data_dir, 'cnews.train.txt')
test_dir = os.path.join(data_dir, 'cnews.test.txt')
val_dir = os.path.join(data_dir, 'cnews.val.txt')

vocab_dir = os.path.join(data_dir, 'cnews.vocab.txt')
tensorboard_dir = 'tensorboard/textCnn'
save_dir = 'checkpoints/textCnn'

save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(model, config):
    """training text cnn"""
    # 载入训练集和验证集
    start_time = time.time()
    x_train, y_train = process_file(
        train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(
        val_dir, word_to_id, cat_to_id, config.seq_length)
    time_df = get_time_dif(start_time)

    print("load train and eval data done, time：", time_df)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_time = time.time()
    steps = 0
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    improved_str = "*"

    flag = False
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(1, config.num_epochs + 1):
        print("Epoch:", epoch)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            x_batch = Variable(torch.LongTensor(x_batch)).cuda()
            y_batch = Variable(torch.LongTensor(y_batch)).cuda()

            output = model(x_batch)
            y_batch = torch.argmax(y_batch, dim=1)
            loss = F.cross_entropy(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % config.print_per_batch == 0:
                eval_acc = eval(x_val, y_val, model, config)
                corrects = (torch.max(output, 1)[1].view(
                    y_batch.size()).data == y_batch.data).sum()
                train_acc = float(corrects)/config.batch_size
                if eval_acc > best_acc_val:
                    best_acc_val = eval_acc
                    last_improved = steps
                    improved_str = "*"
                else:
                    improved_str = ""
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0},\tTrain Loss: {1:.3},\tTrain Acc: {2:.3}\tVal Acc: {3:.3}, \tTime: {4} {5}'
                print(msg.format(steps, loss.float(),
                                 train_acc, eval_acc, time_dif, improved_str))
            if steps - last_improved > config.require_improvement:
                print("No optimization for a long time, auto-stopping... pytorch")
                flag = True
                break  # 跳出循环
        if flag:
            break
    #
        # TODO: 验证集


def eval(x_val, y_val, model, config):
    """验证model效果"""
    model.eval()
    corrects, avg_loss = 0, 0

    batch_train = batch_iter(x_val, y_val, config.batch_size)
    for x_batch, y_batch in batch_train:
        x_batch = Variable(torch.LongTensor(x_batch)).cuda()
        y_batch = Variable(torch.LongTensor(y_batch)).cuda()

        output = model(x_batch)

        y_batch = torch.argmax(y_batch, dim=1)
        loss = F.cross_entropy(output, y_batch)
        avg_loss += loss.data.float()
        corrects += (torch.max(output, 1)
                     [1].view(y_batch.size()).data == y_batch.data).sum()
    size = x_val.shape[0]
    avg_loss /= size
    accuracy = float(corrects)/size
    return accuracy


if __name__ == "__main__":
    """入口函数"""
    task_type = ["train", "test"]
    if len(sys.argv) != 2 or sys.argv[1] not in task_type:
        raise ValueError("usage: python run.py [train/test]....")
    print("config text cnn model by pytorch...")

    config = TextCnnConfig()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCnn(config)

    if torch.cuda.is_available():
        print("cuda is avaiable.....")
        model = model.cuda()
    else:
        print("cuda is not avaiable.....")

    if sys.argv[1] == task_type[0]:
        train(model, config)
    elif sys.argv[1] == task_type[1]:
        # eval()
        pass
