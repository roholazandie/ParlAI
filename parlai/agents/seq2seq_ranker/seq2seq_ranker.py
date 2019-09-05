from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.agents.seq2seq.modules import Seq2seq, opt_to_kwargs
from parlai.core.utils import warn_once

import torch
import torch.nn as nn


class Seq2seqRankerAgent(TorchRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot',
                                    'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')

        super(Seq2seqRankerAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'Seq2Seq'

    def build_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        self.model = Seq2seq(
            len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            end_idx=self.END_IDX, unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(self.model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'])

        if self.use_cuda:
            self.model.cuda()

        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False

        if self.use_cuda:
            self.model.cuda()

        return self.model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)

        if self.use_cuda:
            self.criterion.cuda()

    # def _score(self, output, cands):
    #     if cands.dim() == 2:
    #         return torch.matmul(output, cands.t())
    #     elif cands.dim() == 3:
    #         return torch.bmm(output.unsqueeze(1),
    #                          cands.transpose(1, 2)).squeeze(1)
    #     else:
    #         raise RuntimeError('Unexpected candidate dimensions {}'
    #                            ''.format(cands.dim()))
    #
    #
    # def compute_loss(self, batch, return_output=False):
    #
    #     if batch.label_vec is None:
    #         raise ValueError('Cannot compute loss without a label.')
    #     #we should replace ys with candidates
    #     model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
    #     scores, preds, *_ = model_output


    def encode_candidates(self, padded_cands):
        pass
        #return self.model.answer_embedder(padded_cands)

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        model_output = self.model(*(batch.text_vec,), ys=cand_vecs)
        scores, preds, *_ = model_output
        #scores = scores.unsqueeze(0)
        # mems = self._build_mems(batch.memory_vecs)
        # # Check for rows that have no non-null tokens
        # pad_mask = None
        # if mems is not None:
        #     pad_mask = (mems != self.NULL_IDX).sum(dim=-1) == 0
        #
        # if cand_encs is not None:
        #     state, _ = self.model(batch.text_vec, mems, None, pad_mask)
        # else:
        #     state, cand_encs = self.model(batch.text_vec, mems, cand_vecs,
        #                                   pad_mask)
        # scores = self._score(state, cand_encs)
        #
        return scores


    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def state_dict(self):
        """Get the model states for saving. Overriden to include longest_label"""
        states = super().state_dict()
        if hasattr(self.model, 'module'):
            states['longest_label'] = self.model.module.longest_label
        else:
            states['longest_label'] = self.model.longest_label

        return states


    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def is_valid(self, obs):
        normally_valid = super().is_valid(obs)
        if not normally_valid:
            # shortcut boolean evaluation
            return normally_valid
        contains_empties = obs['text_vec'].shape[0] == 0
        if self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) during training. '
                'Skipping this example, but you should check your dataset and '
                'preprocessing.'
            )
        elif not self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) in an '
                'evaluation example! This may affect your metrics!'
            )
        return not contains_empties
