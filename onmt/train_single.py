#!/usr/bin/env python
"""Training on a single process."""
import os

import torch
import pdb
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.translate.translator import train_build_translator
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
#     import pdb
#     _check_ = torch.load("/home/irteam/users/kaist/ginalee/clean_data/baselines/9-domain5-185pre_step_2500.pt")
#     model_encoder = [i for i in _check_['model'].keys() if "encoder" in i.split(".")]
#     encoder = {}
#     pdb.set_trace()
#     for i, param in enumerate(model_encoder):
#         if i == 0:
#             encoder['embeddings.word_embeddings.weight'] = _check_['model'][param]
#             continue
#         param_ = ".".join(param.split(".")[1:])
# #         if param.split(".")[1] == 'encoder':
# #             param_ = ".".join(param.split(".")[2:])
# #         else:
# #             param_ = ".".join(param.split(".")[1:])
#         encoder[param_] = _check_['model'][param]
#     pdb.set_trace()
    
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    logger.info(opt)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)

        
        load_vocab = torch.load(opt.data + '.vocab.pt')
        vocab = checkpoint['vocab']
        load_vocab['src'].fields[0][1].vocab = vocab['src'].fields[0][1].vocab
        load_vocab['tgt'].fields[0][1].vocab = vocab['tgt'].fields[0][1].vocab
        vocab = load_vocab
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')
    
    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    if opt.pretrain_from:
        check = torch.load(opt.pretrain_from,
                            map_location=lambda storage, loc: storage)
        model.load_state_dict(check['model'],strict=False)
        model.load_state_dict(check['generator'],strict=False)
        if 'dom_classifier' in check:
            model.load_state_dict(check['dom_classifier'],strict=False)

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)
    
    translator = None
    if opt.domain_cls_enc == False:
        translator = train_build_translator(opt, model, model_opt, fields, report_score=True)

    trainer = build_trainer(
        translator, opt, device_id, model, fields, optim, model_saver=model_saver)

    train_iter = build_dataset_iter("train", fields, opt)
    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
