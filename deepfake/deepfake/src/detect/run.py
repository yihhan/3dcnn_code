import argparse
import logging
import pickle
import numpy as np
import torch
import yaml
import os, os.path as osp

try:
    from .factory import set_reproducibility
    from .factory import train as factory_train
    from .factory import evaluate as factory_evaluate
    from .factory import builder 
except:
    from factory import set_reproducibility
    import factory.train as factory_train
    import factory.evaluate as factory_evaluate
    import factory.builder as builder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=-1)
    return parser.parse_args()

def create_logger(cfg, mode):
    logfile = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], 'log_{}.txt'.format(mode))
    if osp.exists(logfile): os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    return logger

def set_inference_batch_size(cfg):
    if 'evaluation' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['evaluation'].keys():
            cfg['evaluation']['batch_size'] = 2*cfg['train']['batch_size']

    if 'test' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['test'].keys():
            cfg['test']['batch_size'] = 2*cfg['train']['batch_size']

    if 'predict' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['predict'].keys():
            cfg['predict']['batch_size'] = 2*cfg['train']['batch_size']

    return cfg 

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.num_workers > 0:
        cfg['transform']['num_workers'] = args.num_workers

    cfg = set_inference_batch_size(cfg)

    # We will set all the seeds we can, in vain ...
    set_reproducibility(cfg['seed'])
    # Set GPU
    torch.cuda.set_device(args.gpu)

    # Make directory to save checkpoints
    if not osp.exists(cfg['evaluation']['params']['save_checkpoint_dir']): 
        os.makedirs(cfg['evaluation']['params']['save_checkpoint_dir'])

    # Load in labels with CV splits
    with open(cfg['dataset']['pickled'], 'rb') as f:
        ann = pickle.load(f)

    ofold = cfg['dataset']['outer_fold']
    ifold = cfg['dataset']['inner_fold']

    if args.mode != 'predict':
        train_ann, valid_ann, test_ann = get_train_valid_test(cfg, ann, ofold, ifold, args.mode)

    logger = create_logger(cfg, args.mode)
    logger.info('Saving to {} ...'.format(cfg['evaluation']['params']['save_checkpoint_dir']))

    if args.mode == 'find_lr':
        cfg['optimizer']['params']['lr'] = cfg['find_lr']['params']['start_lr']
        if 'grad_clip' in cfg['train']['params'].keys():
            cfg['find_lr']['params']['grad_clip'] = cfg['train']['params']['grad_clip']
        find_lr(args, cfg, train_ann, valid_ann)
    elif args.mode == 'train':
        train(args, cfg, train_ann, valid_ann)
    elif args.mode == 'predict':
        predict(args, cfg)

def get_train_valid_test(cfg, ann, ofold, ifold, mode):
    if ofold == 'valid' or ifold == 'valid': 
        # This means simple train/valid split
        train_ann = [a for a in ann if a['split'] == 'train']
        valid_ann = [a for a in ann if a['split'] == 'valid']
        test_ann  = [a for a in ann if a['split'] == 'valid']
        return train_ann, valid_ann, test_ann
    # Get train/validation set
    if cfg[mode.replace('find_lr','train')]['outer_only']: 
        # valid and test are essentially the same here
        train_ann = [a for a in ann if a['cv_splits']['outer'] != ofold]
        valid_ann = [a for a in ann if a['cv_splits']['outer'] == ofold]
        test_ann  = [a for a in ann if a['cv_splits']['outer'] == ofold]
    else:
        test_ann = [a for a in ann if a['cv_splits']['outer'] == ofold]
        ann = [a for a in ann if a['cv_splits']['outer'] != ofold]
        train_ann = [a for a in ann if a['cv_splits']['inner{}'.format(ofold)] != ifold]
        valid_ann = [a for a in ann if a['cv_splits']['inner{}'.format(ofold)] == ifold]
    return train_ann, valid_ann, test_ann


def setup(args, cfg, train_ann, valid_ann):

    logger = logging.getLogger('root')

    train_loader = builder.build_dataloader(cfg, ann=train_ann, mode='train')
    valid_loader = builder.build_dataloader(cfg, ann=valid_ann, mode='valid')
    
    # Adjust steps per epoch if necessary (i.e., equal to 0)
    # We assume if gradient accumulation is specified, then the user
    # has already adjusted the steps_per_epoch accordingly in the 
    # config file
    steps_per_epoch = cfg['train']['params']['steps_per_epoch']
    gradient_accmul = cfg['train']['params']['gradient_accumulation']
    if steps_per_epoch == 0:
        cfg['train']['params']['steps_per_epoch'] = len(train_loader)

    logger.info('Building [{}] architecture ...'.format(cfg['model']['config']))
    model = builder.build_model(cfg, args.gpu)
    model = model.train().cuda()

    optimizer = builder.build_optimizer(
        cfg['optimizer']['name'], 
        model, 
        cfg['optimizer']['params'])
    scheduler = builder.build_scheduler(
        cfg['scheduler']['name'], 
        optimizer, 
        cfg=cfg)

    return cfg, \
           train_loader, \
           valid_loader, \
           model, \
           optimizer, \
           scheduler 

def find_lr(args, cfg, train_ann, valid_ann):

    logger = logging.getLogger('root')

    logger.info('FINDING LR ...')

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    scheduler = setup(args, cfg, train_ann, valid_ann)

    finder = factory_train.LRFinder(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        save_checkpoint_dir=cfg['evaluation']['params']['save_checkpoint_dir'],
        logger=logger,
        gradient_accumulation=cfg['train']['params']['gradient_accumulation'])

    finder.find_lr(**cfg['find_lr']['params'])

    logger.info('Results are saved in : {}'.format(osp.join(finder.save_checkpoint_dir, 'lrfind.csv')))

def train(args, cfg, train_ann, valid_ann):
    
    logger = logging.getLogger('root')

    logger.info('TRAINING : START')

    logger.info('TRAIN: n={}'.format(len(train_ann)))
    logger.info('VALID: n={}'.format(len(valid_ann)))

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    scheduler = setup(args, cfg, train_ann, valid_ann)

    if 'load_previous' in cfg['train'].keys() and cfg['train']['load_previous']:
        logger.info ('Loading previous checkpoint from {} ...'.format(cfg['train']['load_previous']))
        model.load_state_dict(torch.load(cfg['train']['load_previous'], map_location=lambda storage, loc: storage))

    evaluator = getattr(factory_evaluate, cfg['evaluation']['evaluator'])
    evaluator = evaluator(dataset=valid_loader.dataset, 
        **cfg['evaluation']['params'])

    trainer = getattr(factory_train, cfg['train']['trainer'])
    trainer = trainer(loader=train_loader,
        model=model,
        optimizer=optimizer,
        schedule=scheduler,
        evaluator=evaluator,
        logger=logger)
    trainer.train(**cfg['train']['params'])


def predict(args, cfg):

    logger = logging.getLogger('root')

    if 'data_dir' in cfg['predict'].keys():
        cfg['dataset']['data_dir'] = cfg['predict']['data_dir']

    with open(cfg['predict']['pickled'], 'rb') as f:
        ann = pickle.load(f)

    logger.info('PREDICT : START')
    logger.info('n={}'.format(len(ann)))

    loader = builder.build_dataloader(cfg, ann=ann, mode='predict')

    logger.info('Building [{}] architecture ...'.format(cfg['model']['config']))
    model = builder.build_model(cfg, args.gpu)
    model.load_state_dict(torch.load(cfg['predict']['checkpoint'], map_location=lambda storage, loc: storage))
    model = model.eval().cuda()

    predictor = getattr(factory_evaluate, cfg['predict']['predictor'])
    predictor = predictor(dataset=loader.dataset, 
                          **cfg['predict']['params'])

    _, y_pred, names = predictor.predict(model=model, epoch=None)

    if not os.path.exists(cfg['predict']['save_preds_dir']):
        os.makedirs(cfg['predict']['save_preds_dir'])

    savefile = osp.join(cfg['predict']['save_preds_dir'], 'predictions.pkl')

    with open(savefile, 'wb') as f:
        pickle.dump({'y_pred': y_pred, 'names': names}, f)


if __name__ == '__main__':
    main()












