""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
import pdb

def user_embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, tgt, opt):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt = tgt
        self.opt = opt
        if opt.user_bias!='none' and opt.user_bias=='full_bias':
            self.user_bias = user_embedding(opt.user_len, opt.voc_len, padding_idx=None)
        elif opt.user_bias!='none' and opt.user_bias=='factor_cell':
            self.user_bias = user_embedding(opt.user_len, opt.fact_len, padding_idx=None)
            self.user_global = nn.Parameter(torch.rand(opt.fact_len, opt.voc_len))
            nn.init.normal_(self.user_global, mean=0, std=opt.voc_len ** -0.5)

    def forward(self, src, tgt, uid, lengths, bptt=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        cls_state= None
        if self.opt.user_cls or self.opt.dom_cls:
            cls = torch.LongTensor([101,0]).unsqueeze(0).unsqueeze(0).expand(-1,src.shape[1],-1).cuda()
            src = torch.cat((cls,src),0)
            pre_enc_state, memory_bank, lengths = self.encoder(src, uid, lengths, **kwargs)
            enc_state = pre_enc_state[:,1:,:]
            cls_state = pre_enc_state[:,0,:]
        else:
            if self.opt.encoder_freeze:
                with torch.no_grad():
                    enc_state, memory_bank, lengths = self.encoder(src, uid, lengths, **kwargs)
            else:
                    enc_state, memory_bank, lengths = self.encoder(src, uid, lengths, **kwargs)                
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)

        if self.opt.user_cls == False:
            dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths, uid=uid)
        else:
            with torch.no_grad():
                dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths)       
        if self.opt.dom_avg_pool:
            return dec_out, attns, enc_state
        
        return dec_out, attns, cls_state

class Domain_CLS_ENC(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, opt):
        super(Domain_CLS_ENC, self).__init__()
        self.encoder = encoder
        self.opt = opt

    def forward(self, src, lengths, bptt=False, **kwargs):

        cls_state= None

        cls = torch.LongTensor([101,0]).unsqueeze(0).unsqueeze(0).expand(-1,src.shape[1],-1).cuda()
        src = torch.cat((cls,src),0)
        pre_enc_state, memory_bank, lengths = self.encoder(src, None, lengths, **kwargs)

        enc_state = pre_enc_state[:,1:,:]
        cls_state = pre_enc_state[:,0,:]        

        return None, None, cls_state
 
