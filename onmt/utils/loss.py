"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from onmt.inputters.text_dataset import MyBertTokenizer
from torchtext.data.metrics import bleu_score
import os

import sacrebleu

import pdb

def build_loss_compute(translator, model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]
    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
        )
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    loss_cls = None
    loss_dom = None
    if opt.user_classify:
        loss_cls = model.classifier
    if opt.domain_classify:
        loss_dom = model.dom_classifier
    if opt.user_bias == "full_bias":
        user_vec = model.user_bias
        glob_vec = None
    elif opt.user_bias == "factor_cell":
        user_vec = model.user_bias
        glob_vec = model.user_global
    else:
        user_vec = None
        glob_vec= None
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength
        )
    else:
        if train:
            compute = NMTLossCompute(None, tgt_field, criterion, loss_gen, opt, user_vec=user_vec, glob_vec=glob_vec, device=device, classification=loss_cls, domain_classification=loss_dom)
        else:
            bert_tok = MyBertTokenizer.from_pretrained('bert-base-multilingual-cased',do_lower_case=False)
            compute = NMTLossCompute(translator, tgt_field, criterion, loss_gen, opt, user_vec=user_vec, glob_vec=glob_vec, bert_tok=bert_tok, device=device, classification=loss_cls, domain_classification=loss_dom)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    # sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator, user_vec=None, glob_vec=None, bert_tok=None, classification=None, domain_classification=None):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.user_vec = user_vec
        self.glob_vec = glob_vec
        self.bert_tok = bert_tok
        self.classification = classification
        self.domain_classification = domain_classification

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def _compute_bleu(self, output, target, **kwargs):
        """
        Compute the bleu. Subclass must define this method.

        Args:

            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 cls_states,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, self.user_vec, self.glob_vec, cls_states, **shard_state)
            if self.bert_tok is not None and output is not None:
                bleu_stats, ter_stats = self._compute_bleu_ter(batch, self.user_vec, self.glob_vec, shard_state['output'], shard_state['target'])
                stats.update_bleu(bleu_stats)
                stats.update_ter(ter_stats)
            return loss / float(normalization), stats
        
        batch_stats = onmt.utils.Statistics()
        if self.bert_tok is not None and output is not None:
            bleu_stats, ter_stats = self._compute_bleu_ter(batch, self.user_vec, self.glob_vec, shard_state['output'], shard_state['target'])
            batch_stats.update_bleu(bleu_stats)
            stats.update_ter(ter_stats)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, self.user_vec, self.glob_vec, cls_states, **shard)
            loss.div(float(normalization)).backward(retain_graph=True)
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, translator, tgt_field, criterion, generator, opts, user_vec=None, glob_vec=None, normalization="sents", bert_tok=None, device=None, classification=None, domain_classification=None):
        super(NMTLossCompute, self).__init__(criterion, generator, user_vec, glob_vec, bert_tok, classification, domain_classification)
        self.opts = opts
        self.device = device
        self.translator = translator
        self.tgt_field = tgt_field

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output if output is not None else batch.tgt[range_[0] + 1: range_[1], :, 0],
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            #"user" : batch.uid #user
        }

    def _compute_loss(self, batch, user_vec, glob_vec, cls_states, output, target):
        padding_idx = self.tgt_field.vocab.stoi[self.tgt_field.pad_token]
        
        # Multi-task case with user_bias
        if self.opts.user_bias != 'none':
            sfm = nn.LogSoftmax(dim=-1)
            bottled_output = self._bottle(output)
            out = self.generator(bottled_output).view(output.shape[0],output.shape[1],-1)
            if glob_vec is not None and self.opts.user_bias=='factor_cell':
                u_vec = torch.matmul(user_vec(batch.uid),glob_vec)
                out = out + u_vec
            else:
                out = out + user_vec(batch.uid)

            scores = out.view(out.shape[0]*out.shape[1],-1)
            scores = sfm(scores).to(self.device)
        
            if self.opts.domain_classify:
                if self.opts.dom_avg_pool:
                    user_dom = self.domain_classification(cls_states.mean(1))
                else:
                    user_dom = self.domain_classification(cls_states)
        
        # Domain (Encoder only) 
        elif self.opts.domain_cls_enc:
            pdb.set_trace()
            user_dom = self.domain_classification(cls_states)
        
        # Multi-task case with APE
        else:           
            bottled_output = self._bottle(output)
            scores = self.generator(bottled_output)
            if self.opts.user_classify:
                if self.opts.user_cls:
                    #pdb.set_trace()
                    user_cls = self.classification(cls_states)
                else:
                    user_cls = self.classification(output.mean(0))
            if self.opts.domain_classify:
                if self.opts.dom_avg_pool:
                    user_dom = self.domain_classification(cls_states.mean(1))
                else:
                    user_dom = self.domain_classification(cls_states)
        
        # Single-task case
        if self.opts.domain_cls_enc:
            dom_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='mean')
            gtruth = batch.dom
            scores = user_dom
            dom_loss = dom_cri(scores, gtruth)
            loss = dom_loss
            
        else:           
            # Decoder LM loss
            gtruth = target.view(-1)
            score_loss = self.criterion(scores, gtruth)
            
            # user classification + scores ( APE or user_bias)
            if self.opts.user_classify and not self.opts.domain_adv:
                cls_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
                user_loss = cls_cri(user_cls, batch.uid)
                scores = user_cls
                gtruth = batch.uid
                #pdb.set_trace()
#                 loss = score_loss+user_loss
                loss = user_loss
                
            # user classification + domain_adv + scores ( APE or user_bias)
            elif self.opts.user_classify and self.opts.domain_adv:
                cls_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
                user_loss = cls_cri(user_cls, batch.uid)
                dom_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
                dom_loss = dom_cri(user_dom, batch.dom)
                loss = score_loss+user_loss-dom_loss
            
            # domain + scores ( APE or user_bias)
            elif self.opts.domain_classify and not self.opts.domain_adv:
                dom_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
                dom_loss = dom_cri(user_dom, batch.dom)
                loss = score_loss+dom_loss
                
            # domain_adv + scores ( APE or user_bias) manipulated by domain_ratio 
            elif self.opts.domain_adv:
                dom_cri = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
                dom_loss = dom_cri(user_dom, batch.dom)
                loss = (1 - self.opts.domain_ratio) * score_loss - (self.opts.domain_ratio) * dom_loss
            
            # scores (APE or user_bias) OR domain_cls enc only
            else:
                loss = score_loss
        
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats
    
    def _build_target_tokens(self, pred):
        
        vocab = self.tgt_field.vocab
        tokens = []
        
        for tok in pred:
            tokens.append(vocab.itos[tok])
            if tokens[-1] == self.tgt_field.eos_token:
                tokens = tokens[:-1]
                break
                
        return tokens
    
    def from_batch(self, translation_batch):
        
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        translations = []
        for b in range(batch_size):

            pred_sents = [self._build_target_tokens(
                translation_batch["predictions"][b][n])
                for n in range(self.opts.n_best)]

            translations.append(pred_sents)

        return translations

    def _compute_bleu_ter(self, batch, user_vec, glob_vec, output, target):
        vocab = self.bert_tok.ids_to_tokens
        
        batch_data = self.translator.translate_batch(batch,vocab,False)
        translations = self.from_batch(batch_data)
        
        sys=[]
        for trans in translations:
            line = ' '.join(trans[0]).replace(' ##', '').replace('##', '')
            sys.append(line)
        
        references = []
        for tgt in batch.tgt.squeeze(-1).transpose(0,1):
            references.append(self._build_target_tokens(tgt))
        
        refs=[]
        for ref in references:
            line = ' '.join(ref[1:]).replace(' ##', '').replace('##', '')
            refs.append(line)
        
        bleu = sacrebleu.corpus_bleu(sys, [refs], force=True)
        ter = sacrebleu.corpus_ter(sys, [refs])

        bleu_stats = onmt.utils.Statistics(bleu=bleu.score * len(sys), sent=len(sys))
        ter_stats = onmt.utils.Statistics(ter=ter.score * len(sys), sent=len(sys))

        return (bleu_stats, ter_stats)


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        if len(variables) > 0:
            inputs, grads = zip(*variables)   
            torch.autograd.backward(inputs[0], grads[0])
