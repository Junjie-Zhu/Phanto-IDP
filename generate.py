import sys, time, csv, os, random, math, argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
from arguments import buildParser

from model import ProteinGCN
from traj_dataset import ProteinDataset, splitDataset, collate_pool
from utils import AverageMeter, Normalizer, Logger, count_parameters, randomSeed, clearCache
import config as cfg


def main():
    global args, savepath, dataset

    parser = buildParser()
    args = parser.parse_args()

    print('Torch Device being used: ', cfg.device)
    print('cuda version: ', torch.version.cuda)
    print('cuda availability: ', torch.cuda.is_available())
     
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
    model = ProteinGCN(**kwargs)

    model = torch.nn.DataParallel(model)
    model.cuda()
    print(model)
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
    
    # generate new conformations based on test set
    if args.save_checkpoints and len(test_loader):
        print('---------Generation Based on Test Set---------------')
        test_state = generate(test_loader, model)


def generate(data_loader, model):
    for protein_batch_iter, (input_data, target_data) in enumerate(data_loader):
        batch_size = len(target_data[0])

        # move inputs and targets to cuda
        input_var, target_var = getInputs(input_data, target_data)

        with torch.no_grad():
            # Switch to evaluation mode
            model.eval()
            predicted = model(input_var)
            for i in range(args.avg_sample):
                conf_seed = model.module.reparameterize(predicted[1], 
                                                        torch.ones(predicted[2].shape).to(predicted[2].device),
                                                        temp=0.05)
                generated = model.module.sample(conf_seed)
                for j in range(generated.shape[0]):
                    np.savetxt("generates/predicted.%d.%d.%d.dat" % (protein_batch_iter, i, j), generated[j].cpu().numpy())

    return 0


def getInputs(inputs, target_tuples):
    """Move inputs and targets to cuda"""

    input_var = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()]
    target_var = [target_tuples[0].cuda(), target_tuples[1].cuda(), target_tuples[2].cuda()]

    return input_var, target_var


if __name__ == '__main__':
    start = time.time()
    main()
    print('Time taken: ', time.time() - start)
