import time
import os.path as osp
import numpy as np
import tqdm
import math

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer, get_dataloader_spl
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from collections import deque
from tqdm import tqdm


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    def train(self):
        args = self.args
        label, label_aux = self.prepare_label()

        # decide lambda0
        print('# compute losses of episodes in a random epoch to init lambda0.')
        loss_list = []
        self.model.eval()
        if args.lambda_spl is not None:
            lambda0 = args.lambda_spl
        else:
            with torch.no_grad():
                for _, batch in enumerate(tqdm(self.train_loader), 1):
                    if torch.cuda.is_available():
                        data, gt_label = [_.cuda() for _ in batch]
                    else:
                        data, gt_label = batch[0], batch[1]
                    logits = self.para_model(data)
                    total_loss = F.cross_entropy(logits, label)
                    loss_list.append(total_loss.item())

                loss_list_sorted = sorted(loss_list)
                lambda0 = loss_list_sorted[math.ceil(len(loss_list) * args.percent)]

        print('\n# init lambda: ' + str(lambda0))

        # start FSL training
        total_tasks = 0
        for epoch in range(1, args.max_epoch + 1):
            print('\n!!!!!! starting epoch %d !!!!!!' % epoch)
            print('# current lambda: ' + str(lambda0))
            self.train_epoch += 1

            classes0 = []
            loss_list = []
            print('# loss compute ...')
            self.model.eval()
            # self.model.encoder.eval()
            with torch.no_grad():
                for _, batch in enumerate(tqdm(self.train_loader), 1):
                    if torch.cuda.is_available():
                        data, gt_label = [_.cuda() for _ in batch]
                    else:
                        data, gt_label = batch[0], batch[1]
                    logits = self.para_model(data)
                    total_loss = F.cross_entropy(logits, label)

                    loss_list.append(total_loss.item())
                    classes0.append(gt_label[:args.way])


            print('# model update ...')
            selected_idx = list(np.where(np.array(loss_list) < lambda0)[0])
            classes = []
            selected_w = []
            for si in selected_idx:
                classes.append(classes0[si])
                if args.soft:
                    soft_w = - loss_list[si] / lambda0 + 1
                    selected_w.append(soft_w)
            train_loader_selected = get_dataloader_spl(args, classes)


            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            for b, batch in enumerate(tqdm(train_loader_selected), 1):
                # if torch.cuda.is_available():
                #     data, gt_label = [_.cuda() for _ in batch]
                # else:
                #     data, gt_label = batch[0], batch[1]
                # logits, reg_logits = self.para_model(data)
                # if reg_logits is not None:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                # else:
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = loss
                # if args.soft:
                #     total_loss = selected_w[b-1] * total_loss

                self.train_step += 1

                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data, _ = batch[0], batch[1]

                # construct pseudo tasks.
                task_groups = []
                task_len = int(len(data)/2)
                task1, task2 = data.split(task_len)
                task1_support, task1_query = task1[:args.way * args.shot], task1[args.way * args.shot:]
                task2_support, task2_query = task2[:args.way * args.shot], task2[args.way * args.shot:]

                # c1
                if args.case == 'c1':
                    task_group1 = [task1, torch.cat([task1_support, task2_query])]
                    task_group2 = [task2, torch.cat([task2_support, task1_query])]
                    task_groups = [task_group1, task_group2]
                # c2
                if args.case == 'c2':
                    task_group3 = [task1, torch.cat([task2_support, task1_query])]
                    task_group4 = [task2, torch.cat([task1_support, task2_query])]
                    task_groups = [task_group3, task_group4]

                for tg in task_groups:
                    ce_loss = 0
                    logits_all = []
                    for i in [0, 1]:
                        logits, reg_logits = self.model(tg[i])
                        if reg_logits is not None:
                            loss = F.cross_entropy(logits, label) + args.balance * F.cross_entropy(reg_logits, label_aux)
                        else:
                            loss = F.cross_entropy(logits, label)
                        ce_loss = ce_loss + loss
                        logits_all.append(logits)

                    q = args.way * args.query
                    kd_pq = F.kl_div(F.log_softmax(logits_all[0], dim=1), F.softmax(logits_all[1].detach(), dim=1), reduction='sum') / q
                    kd_qp = F.kl_div(F.log_softmax(logits_all[1], dim=1), F.softmax(logits_all[0].detach(), dim=1), reduction='sum') / q

                    kl_loss = (kd_pq + kd_qp) / 2

                    # total_loss = (1-args.balance_kl) * ce_loss_all + args.balance_kl * kl_loss_all
                    total_loss = ce_loss + kl_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                total_tasks = total_tasks + 1

            self.lr_scheduler.step()
            # update lambda0
            lambda0 = lambda0 + args.inc
            print('total tasks: ' + str(total_tasks))

            # if epoch < 30:
            #     continue
            self.try_evaluate(epoch)

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')


    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation model
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader), 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        # train model
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap


    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation model
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((args.num_test_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('\nbest val_acc={:.4f} \n'.format(self.trlog['max_acc']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']))

        return vl, va, vap


    def final_record(self):
        # save the best performance in a txt file

        with open(
                osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])),
                'w') as f:
            f.write('best val_loss={:.4f} \n'.format(self.trlog['min_loss']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))