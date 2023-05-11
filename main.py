import argparse
import math
import os
import random
import time

import numpy as np
import torch

import config as cfg
from arguments import buildParser
from model import PhantoIDP
from traj_dataset import ProteinDataset, splitDataset, collate_pool
from utils import AverageMeter, count_parameters, randomSeed, clearCache


def main():
    global args, savepath, dataset

    parser = buildParser()
    args = parser.parse_args()

    print('Torch Device being used: ', cfg.device)
    print('cuda version: ', torch.version.cuda)
    print('cuda availability: ', torch.cuda.is_available())
    # sys.exit(0)

    # create the savepath
    savepath = args.save_dir + str(args.name) + '/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    randomSeed(args.seed)

    # create train/val/test dataset separately
    assert os.path.exists(args.protein_dir), '{} does not exist!'.format(args.protein_dir)
    all_dirs = [d for d in os.listdir(args.protein_dir) if not d.startswith('.DS_Store')]
    dir_len = len(all_dirs)
    indices = list(range(dir_len))
    random.shuffle(indices)

    train_size = math.floor(args.train * dir_len)
    val_size = math.floor(args.val * dir_len)
    test_size = math.floor(args.test * dir_len)

    if val_size == 0:
        print(
            'No protein directory given for validation!! Please recheck the split ratios, ignore if this is intended.')
    if test_size == 0:
        print('No protein directory given for testing!! Please recheck the split ratios, ignore if this is intended.')

    test_dirs = all_dirs[:test_size]
    train_dirs = all_dirs[test_size:test_size + train_size]
    val_dirs = all_dirs[test_size + train_size:test_size + train_size + val_size]
    print('Testing on {} protein directories:'.format(len(test_dirs)))

    dataset = ProteinDataset(args.pkl_dir, args.protein_dir, args.atom_init, random_seed=args.seed)

    print('Dataset length: ', len(dataset))

    # load all model args from pretrained model
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print("=> loading model params '{}'".format(args.pretrained))
        model_checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        # override all args value with model_args
        args.h_a = model_args.h_a
        args.h_g = model_args.h_g
        args.n_conv = model_args.n_conv
        args.random_seed = model_args.seed
        args.lr = model_args.lr

        print("=> loaded model params '{}'".format(args.pretrained))
    else:
        print("=> no model params found at '{}'".format(args.pretrained))

    # build model
    kwargs = {
        'pkl_dir': args.pkl_dir,  # Root directory for data
        'atom_init': args.atom_init,  # Atom Init filename
        'h_a': args.h_a,  # Dim of the hidden atom embedding learnt
        'h_g': args.h_g,  # Dim of the hidden graph embedding after pooling
        'n_conv': args.n_conv,  # Number of GCN layers

        'random_seed': args.seed,  # Seed to fix the simulation
        'lr': args.lr,  # Learning rate for optimizer
    }

    structures, _, _ = dataset[0]
    h_b = structures[1].shape[-1]
    kwargs['h_b'] = h_b  # Dim of the bond embedding initialization

    # Use DataParallel for faster training
    print("Let's use", torch.cuda.device_count(), "GPUs and Data Parallel Model.")
    model = PhantoIDP(**kwargs)

    print(model)
    model = torch.nn.DataParallel(model)
    model.cuda()
    print("model at gpu/%s" % next(model.parameters()).device)

    print('Trainable Model Parameters: ', count_parameters(model))

    # Create dataloader to iterate through the dataset in batches
    train_loader, val_loader, test_loader = splitDataset(dataset, train_dirs, val_dirs, test_dirs,
                                                         collate_fn=collate_pool,
                                                         num_workers=args.workers,
                                                         batch_size=args.batch_size,
                                                         pin_memory=False)

    try:
        print('Training data    : ', len(train_loader.sampler))
        print('Validation data  : ', len(val_loader.sampler))
        print('Testing data     : ', len(test_loader.sampler))
    except Exception as e:
        # sometimes test may not be defined
        print('\nException Cause: {}'.format(e.args[0]))

    # load the model state dict from given pretrained model
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print("=> loading model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)

        model.module.load_state_dict(checkpoint['state_dict'])
        model.module.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no model found at '{}'".format(args.pretrained))

    if not args.no_train:
        is_best = False
        best_val_state = 1e10
        # Main training loop
        for epoch in range(args.epochs):
            # Training
            train_state = trainModel(train_loader, model, epoch=epoch)
            # Validation
            val_state = trainModel(val_loader, model, epoch=epoch, evaluation=True)

            is_best = val_state < best_val_state
            best_val_state = min(val_state, best_val_state)

            # save best model
            if args.save_checkpoints:
                model.module.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': model.module.optimizer.state_dict(),
                    'args': vars(args)
                }, is_best, savepath)

    # test best model using saved checkpoints
    if args.save_checkpoints and len(test_loader):
        print('---------Evaluate Model on Test Set---------------')
        # this try/except allows the code to test on the go or by defining a pretrained path separately
        try:
            best_checkpoint = torch.load(savepath + 'model_best.pth.tar')
        except Exception as e:
            best_checkpoint = torch.load(args.pretrained)

        model.module.load_state_dict(best_checkpoint['state_dict'])
        test_state = trainModel(test_loader, model, testing=True)


def trainModel(data_loader, model, epoch=None, evaluation=False, testing=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kl_loss = AverageMeter()
    rmsd, batch_idx = 0, 1
    test_rmsd = []

    # weight for FAPE and KL loss
    if epoch != None:
        weight_list = [1e-4, 5e-4, 1e-3, 2.5e-3, 7.5e-3, 1e-2, 1.5e-2]
        weight_fape_list = [10.0, 2.0, 1.0]
        weight = (weight_fape_list[min(epoch // 400, 2)], weight_list[min(epoch // 60, 6)])
    else:
        weight = (10.0, 1e-2)

    end = time.time()

    for protein_batch_iter, (input_data, target_data) in enumerate(data_loader):
        batch_size = len(target_data[0])

        # measure data loading time
        data_time.update(time.time() - end)

        # move inputs and targets to cuda
        input_var, target_var = getInputs(input_data, target_data)

        if not evaluation and not testing:
            # Switch to train mode
            model.train()
            out = model(input_var)
            assert out[0].shape[1] == target_var[0].shape[
                1], "Predicted Outputs Amino & Target Outputs Amino don't match"
            model.module.fit(out, target_var, weight)
        else:
            # evaluate one iteration
            with torch.no_grad():
                # Switch to evaluation mode
                model.eval()
                predicted = model(input_var)
                assert predicted[0].shape[1] == target_var[0].shape[
                    1], "Predicted Outputs Amino & Target Outputs Amino don't match"
                model.module.fit(predicted, target_var, weight, pred=True)

                if testing:
                    targets_val = torch.stack(target_var, dim=-2)
                    targets_val = targets_val.reshape(batch_size, -1, 3)
                    outputs_val = predicted[0].reshape(batch_size, -1, 3)

                    test_rmsd.extend(model.module.calc_rmsd(targets_val, outputs_val))
                    rmsd += test_rmsd[-1]
                    batch_idx += 1

                    for i in range(outputs_val.shape[0]):
                        np.savetxt("outputs/target.%d.%d.dat" % (protein_batch_iter, i), targets_val[i].cpu().numpy())
                        np.savetxt("outputs/predicted.%d.%d.dat" % (protein_batch_iter, i),
                                   outputs_val[i].cpu().numpy())

        # measure accuracy and record loss
        losses.update(model.module.loss.item(), batch_size)
        kl_loss.update(model.module.kl_loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress between steps
        if protein_batch_iter % args.print_freq == 0:
            if evaluation or testing:
                print('Test: [{0}][{1}]/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'KL {kl.val:.4f}\t'.format(
                    epoch, protein_batch_iter, len(data_loader),
                    batch_time=batch_time,
                    loss=losses, kl=kl_loss))
            else:
                print('Epoch: [{0}][{1}]/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'KL {kl.val:.4f}\t'.format(
                    epoch, protein_batch_iter, len(data_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, kl=kl_loss))

        if protein_batch_iter % args.print_freq == 0:
            clearCache()

    # write results to file
    if testing:
        star_label = '**'
        np.savetxt('rmsd.dat', test_rmsd)
    elif evaluation:
        star_label = '*'
    else:
        star_label = '##'

    rmsd /= batch_idx
    print(' {star} LOSS {avg_loss.avg:.3f} KL {kl_loss.avg:.3f}  RMSD {avg_rmsd:.3f}'.format(
        star=star_label, avg_loss=losses, kl_loss=kl_loss, avg_rmsd=rmsd))

    return losses.avg


def getInputs(inputs, target_tuples):
    """Move inputs and targets to cuda"""

    input_var = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()]
    target_var = [target_tuples[0].cuda(), target_tuples[1].cuda(), target_tuples[2].cuda()]

    return input_var, target_var


if __name__ == '__main__':
    start = time.time()
    main()
    print('Time taken: ', time.time() - start)
