import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import resnet18, resnet34, resnet50, vgg16
import Sampling
import datasets
import models
import torchvision.models as torch_models
import pickle
from utils import AverageMeter, Logger
# from center_loss import CenterLoss

from extract_features import CIFAR100_LOAD_ALL

# Create a new ArgumentParser object. ArgumentParser objects hold all the information necessary to parse the command
# line arguments. The description argument gives a brief description of what the program does and it is displayed
# when the -h or --help command line option is given.
"""
parser = argparse.ArgumentParser(description='This is a description of what this program does')
"""

# Add arguments to the parser. Each argument is defined by a name (or names), the type of the argument,
# a default value, and a help message. The 'choices' parameter restricts the argument's value to be one of the
# provided choices.

# The add_argument method is used to specify which command-line options the program is expecting. In the simplest
# form, the add_argument method takes a single argument, a string which specifies the name of the command-line
# option. The type of the argument can also be specified. If not provided, str is the default type. The default value
# of the argument can also be specified. The help message is displayed when the -h or --help command line option is
# given.
"""
parser.add_argument('--argname', type=int, default=0, help='This is a description of what this argument does')
"""

# The parse_args method is called to convert the args at the command line into an object with attributes. If the
# command-line arguments do not match the types and/or constraints defined in the add_argument calls, parse_args will
# raise an error.
"""
args = parser.parse_args()
"""

# The values of the arguments can then be accessed through the args object. For example, if an argument named
# 'argname' was added to the parser, its value can be accessed with 'args.argname'.
"""
print(args.argname)
"""


parser = argparse.ArgumentParser("NEAT")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['Tiny-Imagenet', 'cifar100', 'cifar10', 'mnist'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-query', type=int, default=10)
parser.add_argument('--query-batch', type=int, default=400)
parser.add_argument('--query-strategy', type=str, default='AV_based2',
                    choices=['random', 'uncertainty',
                             'AV_temperature', 'NEAT_passive', 'NEAT',
                             "BGADL", "OpenMax", "Core_set", 'BADGE_sampling', "certainty", "hybrid-BGADL",
                             "hybrid-OpenMax", "hybrid-Core_set", "hybrid-BADGE_sampling", "hybrid-uncertainty"])
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'vgg16'])
# misc
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--init-percent', type=int, default=1)

# active learning

parser.add_argument('--active', action='store_true', help="whether to use active learning")

parser.add_argument('--k', type=int, default=10)

parser.add_argument('--runs', type=int, default=3)

parser.add_argument('--active_5', action='store_true', help="whether to use active learning")

parser.add_argument('--pre-type', type=str, default='clip')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    if args.query_strategy in ['NEAT_passive', 'NEAT', 'hybrid-BGADL', 'hybrid-OpenMax', 'hybrid-Core_set',
                               'hybrid-BADGE_sampling', 'hybrid-uncertainty']:
        ordered_feature, ordered_label, index_to_label = CIFAR100_LOAD_ALL(dataset=args.dataset, pre_type=args.pre_type)

    sys.stdout = Logger(osp.join(args.save_dir, args.query_strategy + '_log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))

    dataset = datasets.create(
        # MISSING PART 1: Provide the name of the dataset
        name="",
        # MISSING PART 2: Provide the known class
        known_class_="",
        # MISSING PART 3: Provide the initial percentage
        init_percent_="",
        # MISSING PART 4: Provide the batch size
        batch_size="",
        # MISSING PART 5: Specify whether to use GPU
        use_gpu="",
        # MISSING PART 6: Provide the number of workers
        num_workers="",
        # MISSING PART 7: Specify whether to filter the dataset
        is_filter="",
        # MISSING PART 8: Specify whether to use a mini version of the dataset
        is_mini="",
        # MISSING PART 9: Provide the seed for random number generation
        SEED="",
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_A, trainloader_B = dataset.trainloader, dataset.trainloader

    negativeloader = None  # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    per_round = []

    per_round.append(list(labeled_ind_train))

    print("Creating model: {}".format(args.model))

    start_time = time.time()

    Acc = {}
    Err = {}
    Precision = {}
    Recall = {}

    for query in tqdm(range(args.max_query)):
        if args.query_strategy in ['NEAT_passive', 'NEAT', 'hybrid-BGADL', 'hybrid-OpenMax', 'hybrid-Core_set',
                                   'hybrid-BADGE_sampling', 'hybrid-uncertainty']:
        # Model initialization
            if args.model == "cnn":
                model = models.create(name=args.model, num_classes=dataset.num_classes)
            elif args.model == "resnet18":
                # 多出的一类用来预测为unknown
                # model_A = resnet18(num_classes=dataset.num_classes + 1)
                model_B = resnet18(num_classes=dataset.num_classes)
            elif args.model == "resnet34":
                # model_A = resnet34(num_classes=dataset.num_classes + 1)
                model_B = resnet34(num_classes=dataset.num_classes)
            elif args.model == "resnet50":
                # model_A = resnet50(num_classes=dataset.num_classes + 1)
                model_B = resnet50(num_classes=dataset.num_classes)

            elif args.model == 'vgg16':
                # model_A = vgg16(num_classes=dataset.num_classes + 1)
                model_B = vgg16(num_classes=dataset.num_classes)

            if use_gpu:
                # model_A = nn.DataParallel(model_A).cuda()
                model_B = nn.DataParallel(model_B).cuda()

            criterion_xent = nn.CrossEntropyLoss()


            optimizer_model_B = torch.optim.SGD(model_B.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)



            if args.stepsize > 0:

                scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)
        else:
            if args.model == "cnn":
                model = models.create(name=args.model, num_classes=dataset.num_classes)
            elif args.model == "resnet18":
                # 多出的一类用来预测为unknown
                model_A = resnet18(num_classes=dataset.num_classes + 1)
                model_B = resnet18(num_classes=dataset.num_classes)
            elif args.model == "resnet34":
                model_A = resnet34(num_classes=dataset.num_classes + 1)
                model_B = resnet34(num_classes=dataset.num_classes)
            elif args.model == "resnet50":
                model_A = resnet50(num_classes=dataset.num_classes + 1)
                model_B = resnet50(num_classes=dataset.num_classes)

            elif args.model == 'vgg16':
                model_A = vgg16(num_classes=dataset.num_classes + 1)
                model_B = vgg16(num_classes=dataset.num_classes)

            if use_gpu:
                model_A = nn.DataParallel(model_A).cuda()
                model_B = nn.DataParallel(model_B).cuda()

            criterion_xent = nn.CrossEntropyLoss()

            optimizer_model_A = torch.optim.SGD(model_A.parameters(), lr=args.lr_model, weight_decay=5e-04,
                                                momentum=0.9)
            optimizer_model_B = torch.optim.SGD(model_B.parameters(), lr=args.lr_model, weight_decay=5e-04,
                                                momentum=0.9)
            # optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

            if args.stepsize > 0:
                scheduler_A = lr_scheduler.StepLR(optimizer_model_A, step_size=args.stepsize, gamma=args.gamma)
                scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)

        # Model training
        for epoch in tqdm(range(args.max_epoch)):
            if args.query_strategy in ['NEAT_passive', 'NEAT', 'hybrid-BGADL', 'hybrid-OpenMax',
                                       'hybrid-Core_set',
                                       'hybrid-BADGE_sampling', 'hybrid-uncertainty']:
                # Train model B for classifying known classes
                train_B(model_B, criterion_xent,
                        optimizer_model_B,
                        trainloader_B, use_gpu, dataset.num_classes, epoch)

                if args.stepsize > 0:
                    # scheduler_A.step()
                    scheduler_B.step()

                if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
                    print("==> Test")
                    # acc_A, err_A = test(model_A, testloader, use_gpu, dataset.num_classes, epoch)
                    acc_B, err_B = test(model_B, testloader, use_gpu, dataset.num_classes, epoch)
                    # print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                    print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))

            else:
                train_A(model_A, criterion_xent,
                        optimizer_model_A,
                        trainloader_A, invalidList, use_gpu, dataset.num_classes, epoch)
                # Train model B for classifying known classes
                train_B(model_B, criterion_xent,
                        optimizer_model_B,
                        trainloader_B, use_gpu, dataset.num_classes, epoch)

                if args.stepsize > 0:
                    scheduler_A.step()
                    scheduler_B.step()

                if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
                    print("==> Test")
                    acc_A, err_A = test(model_A, testloader, use_gpu, dataset.num_classes, epoch)
                    acc_B, err_B = test(model_B, testloader, use_gpu, dataset.num_classes, epoch)
                    print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                    print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))

        # Record results
        acc, err = test(model_B, testloader, use_gpu, dataset.num_classes, args.max_epoch)

        Acc[query], Err[query] = float(acc), float(err)

        # Query samples and calculate precision and recall
        queryIndex = []

        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.random_sampling("", "", "", "", "")
        elif args.query_strategy == "uncertainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.uncertainty_sampling("", "", "", "", "", "")

        elif args.query_strategy == "AV_temperature":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling_temperature(args,
                                                                                                         unlabeledloader,
                                                                                                         len(labeled_ind_train),
                                                                                                         model_A,
                                                                                                         use_gpu)

        elif args.query_strategy == "BGADL":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.bayesian_generative_active_learning(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)
        elif args.query_strategy == "OpenMax":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.openmax_sampling(args, unlabeledloader,
                                                                                                  len(labeled_ind_train),
                                                                                                  model_A, use_gpu,
                                                                                                  openmax_beta=0.5)

        elif args.query_strategy == "Core_set":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.core_set(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)

        elif args.query_strategy == "certainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.certainty_sampling(args, unlabeledloader,
                                                                                                  len(labeled_ind_train), model_A, use_gpu)

        elif args.query_strategy == "BADGE_sampling":

            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.badge_sampling(args, unlabeledloader,

                                                                                    len(labeled_ind_train),

                                                                                    len(unlabeled_ind_train),

                                                                                    labeled_ind_train,

                                                                                    invalidList,

                                                                                    model_A, use_gpu)

        elif args.query_strategy == "NEAT_passive":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.test_query_2(args, model_B, query,
                                                                                              unlabeledloader,
                                                                                              len(labeled_ind_train),
                                                                                              use_gpu,
                                                                                              labeled_ind_train,
                                                                                              invalidList,
                                                                                              unlabeled_ind_train,
                                                                                              ordered_feature,
                                                                                              ordered_label,
                                                                                              index_to_label)


        elif args.query_strategy == "NEAT":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.active_query("", "", "", "", "", "", "", "", "", "", "", "")


        elif args.query_strategy in ["hybrid-BGADL", "hybrid-OpenMax", "hybrid-Core_set", "hybrid-BADGE_sampling", "hybrid-uncertainty"]:
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.passive_and_implement_other_baseline(
                args, model_B, query,
                unlabeledloader,
                len(labeled_ind_train), len(unlabeled_ind_train),
                use_gpu, labeled_ind_train, invalidList, unlabeled_ind_train, ordered_feature, ordered_label,
                index_to_label)

        per_round.append(list(queryIndex) + list(invalidIndex))

        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train) - set(queryIndex))

        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)

        invalidList = list(invalidList) + list(invalidIndex)

        print("Query round: " + str(query) + " | Query Strategy: " + args.query_strategy + " | Query Batch: " + str(
            args.query_batch) + " | Valid Query Nums: " + str(len(queryIndex)) + " | Query Precision: " + str(
            Precision[query]) + " | Query Recall: " + str(Recall[query]) + " | Training Nums: " + str(
            len(labeled_ind_train)) + " | Unalebled Nums: " + str(len(unlabeled_ind_train)))
        if args.query_strategy in ['NEAT_passive', 'NEAT', 'hybrid-BGADL', 'hybrid-OpenMax', 'hybrid-Core_set',
                                   'hybrid-BADGE_sampling', 'hybrid-uncertainty']:
            B_dataset = datasets.create(
                name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
                batch_size=args.batch_size, use_gpu=use_gpu,
                num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
                unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
            )

            trainloader_B, unlabeledloader = B_dataset.trainloader, B_dataset.unlabeledloader
        else:
            dataset = datasets.create(
                name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
                batch_size=args.batch_size, use_gpu=use_gpu,
                num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
                unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train, invalidList=invalidList,
            )

            trainloader_A, testloader = dataset.trainloader, dataset.testloader
            # labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train
            B_dataset = datasets.create(
                name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
                batch_size=args.batch_size, use_gpu=use_gpu,
                num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini, SEED=args.seed,
                unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
            )
            trainloader_B, unlabeledloader = B_dataset.trainloader, B_dataset.unlabeledloader

    #############################################################################################################

    '''
    file_name = "./log_AL/temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type) + "_neighbor_" + str(args.k)
    '''

    file_name = "./log_AL/hybrid_temperature_" + args.model + "_" + args.dataset + "_known" + str(
        args.known_class) + "_init" + str(
        args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
        args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
        args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type)

    ## Save results
    with open(file_name + ".pkl", 'wb') as f:

        data = {'Acc': Acc, 'Err': Err, 'Precision': Precision, 'Recall': Recall}
        pickle.dump(data, f)

    #############################################################################################################
    '''
    selected_index = "./log_AL/temperature_" + args.model + "_" + args.dataset + "_known" + str(args.known_class) + "_init" + str(
                    args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
                    args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
                    args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type) + "_neighbor_" + str(args.k)
    '''

    selected_index = "./log_AL/hybrid_temperature_" + args.model + "_" + args.dataset + "_known" + str(
        args.known_class) + "_init" + str(
        args.init_percent) + "_batch" + str(args.query_batch) + "_seed" + str(
        args.seed) + "_" + args.query_strategy + "_unknown_T" + str(args.unknown_T) + "_known_T" + str(
        args.known_T) + "_modelB_T" + str(args.modelB_T) + "_pretrained_model_" + str(args.pre_type)

    with open(selected_index + "_per_round_query_index.pkl", 'wb') as f:

        pickle.dump(per_round, f)
    #############################################################################################################

    f.close()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def calculate_precision_recall():
    precision, recall = 0, 0
    return precision, recall


def train_A(model, criterion_xent,
            optimizer_model,
            trainloader, invalidList, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()


    invalid_class = args.known_class
    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        features, outputs = model(data)
        loss_xent = criterion_xent(outputs, labels)

        loss_cent = 0.0

        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))





def train_B(model, criterion_xent, optimizer_model, trainloader, use_gpu, num_classes, epoch):
    # Set the model to training mode
    model.train()

    # Initialize the average meter for total loss and cross entropy loss
    xent_losses = AverageMeter()
    losses = AverageMeter()

    # Iterate over the training data
    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        # If GPU is available, move the data and labels to GPU
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        # MISSING PART 1: Get the features and outputs from the model
        ""

        # MISSING PART 2: Compute the cross entropy loss
        ""

        # Initialize the center loss
        loss_cent = 0.0

        # Compute the total loss
        loss = loss_xent + loss_cent

        # MISSING PART 3: Zero the gradients of the optimizer
        ""

        # MISSING PART 4: Backpropagate the loss
        ""

        # MISSING PART 5: Perform a step of the optimizer
        ""

        # Update the losses
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))






def test(model, testloader, use_gpu, num_classes, epoch):
    # Set the model to evaluation mode. This is necessary as some layers like
    # dropout and batch normalization behave differently during training and evaluation.
    model.eval()

    # Initialize counters for correct predictions and total number of data points
    correct, total = 0, 0

    # torch.no_grad() impacts the autograd engine and deactivates it. It reduces memory usage and
    # speeds up computation but you won’t be able to backprop (which you don’t want in an eval script).
    with torch.no_grad():
        # Loop over each batch of data in the test loader
        for index, (data, labels) in testloader:
            # If GPU is available, move the data and labels to GPU
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            # Forward pass: compute the output of the network given the input data
            # MISSING PART 1: Call the model with the input data and get the features and outputs
            "features, outputs = "

            # Get the class with the highest probability from the output
            # MISSING PART 2: Use the outputs to get the predicted class labels
            "predictions = "

            # Update the total number of data points
            total += labels.size(0)

            # Update the number of correct predictions
            # MISSING PART 3: Compare the predicted labels with the true labels and update the 'correct' counter
            "correct += "

    # Compute the accuracy of the model
    acc = correct * 100. / total

    # Compute the error rate of the model
    err = 100. - acc

    # Return the accuracy and error rate
    return acc, err




if __name__ == '__main__':
    main()




