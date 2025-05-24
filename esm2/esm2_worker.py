import functools

import torch
import torch.nn as nn

import esm2
from esm2.logging_utils import get_logger

log = get_logger(__name__)

def make_table(src_alphabet: list[str], esm_alphabet: esm2.Alphabet) -> torch.LongTensor:
    """ Create a translation table from src_alphabet to tgt_alphabet."""
    table = torch.zeros(len(src_alphabet), dtype=torch.long)
    for i, tok in enumerate(src_alphabet):
        # if missing, use the unk token
        table[i] = esm_alphabet.get_idx(tok)
    return table

class ESM2Worker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_iterations = 1

        self._src_alphabet = None
        self._alphabet = None

        self.num_layers = cfg.num_layers
        self.embed_dim = cfg.embed_dim
        self._return_repr_layers = [self.num_layers]
        
        self.load_model() 

    @functools.cached_property
    def translation_table(self) -> torch.LongTensor:
        """ Translate from other token ids to esm2 token ids."""
        if self._src_alphabet is None:
            return
        _table = make_table(self._src_alphabet, self.alphabet)
        return _table

    def set_src_alphabet(self, src_alphabet):
        """ Set the source alphabet for translation."""
        self._src_alphabet = src_alphabet
        # check if we can properly translate
        _table = self.translation_table
        log.info(f"Translation table: {_table}")
    
    def load_model(self):
        self.model, self.alphabet = esm2.pretrained.load_model_and_alphabet_local(self.cfg.ckpt_file)


    @property
    def pad_id(self):
        return self.alphabet.padding_idx

    @property
    def eos_id(self):
        return self.alphabet.eos_idx
    
    @property
    def mask_id(self):
        return self.alphabet.mask_idx

    @functools.cached_property
    def batch_converter(self):
        return self.alphabet.get_batch_converter()

    def model_forward(self, tokens):
        output = self.model(tokens, repr_layers=self._return_repr_layers)
        node_acts = output["representations"][
            self._return_repr_layers[-1]
        ]  # activations from the last layer
        logits = output["logits"]
        pair_acts = None
        return logits, node_acts, pair_acts

    @torch.no_grad()
    def embed_chains(self, inputs, remove_special_tokens=True, **kwargs):
        chain_index = inputs["chain_index"]
        mask = inputs["mask"]
        tokens = inputs["esm_tokens"]  # assume special tokens are added
        init_lengths = _find_lengths(
            tokens, self.pad_id, self.eos_id, use_eos=self.alphabet.append_eos
        )
        tokens = self.trim_node(tokens)
        assert chain_index.shape == tokens.shape, "chain_index and tokens must have the same shape"
        chain_tokens, index = _split_chains(tokens, chain_index, mask)
        # print(tokens, chain_index, mask, index)
        tokens, chain_lengths = _merge_chains(
            chain_tokens, pad_value=self.pad_id
        )  # chain tokens and chain lengths
        # print(chain_lengths)
        eos_position = chain_lengths + int(self.alphabet.prepend_bos) # right shift one if bos is prepended
        tokens = _pad_ends(
            tokens, eos_position, bos_value=self.alphabet.cls_idx, eos_value=self.eos_id, pad_value=self.pad_id
        )
        logits, node_acts, _ = self.model_forward(tokens)

        # trim
        node_acts, logits = map(self.trim_node, (node_acts, logits))
        # gather
        node_acts, logits = map(
            lambda x: _gather_chains(x, chain_lengths, index), (node_acts, logits)
        )
        node_acts = node_acts.masked_fill(~mask[..., None], 0)
        logits = logits.masked_fill(~mask[..., None], 0)

        if not remove_special_tokens:
            # pad
            node_acts, logits = map(
                lambda x: _pad_ends(x, init_lengths, bos_value=0.0, eos_value=0.0, pad_value=0.0),
                (node_acts, logits),
            )

        return logits, node_acts

    @torch.no_grad()
    def embed(self, inputs, split_chains: bool = True, remove_special_tokens=True, **kwargs):
        if split_chains:
            logits, node_acts = self.embed_chains(
                inputs, remove_special_tokens=remove_special_tokens, **kwargs
            )
        else:
            # assume special tokens are added
            logits, node_acts, _ = self.model_forward(inputs["esm_tokens"])
            if remove_special_tokens:
                node_acts, logits = map(self.trim_node, (node_acts, logits))

        # prepare outputs

        out = {
            "esm_pred_logits": logits,
            "esm_node_acts": node_acts,
        }

        if "esm_gt_tokens" in inputs:
            gt_tokens = inputs["esm_gt_tokens"]
            if remove_special_tokens:
                gt_tokens = self.trim_node(gt_tokens)
            out["esm_gt_tokens"] = gt_tokens

        return out

    def trim_node(self, x):
        if x is None:
            return
        return _trim_ends(x, trim_bos=self.alphabet.prepend_bos, trim_eos=self.alphabet.append_eos)

    @torch.no_grad()
    def encode_seqs(self, seqs: list[tuple[str, str]]) -> torch.LongTensor:
        _, _, batch_tokens = self.batch_converter(seqs)
        return batch_tokens
    
    def prepare_inputs(self, src_tokens: torch.LongTensor, src_lengths: torch.LongTensor, pred_mask: torch.BoolTensor):
        src_tokens = src_tokens.long()
        src_lengths = src_lengths.long()
        pred_mask = pred_mask.bool()
        table = self.translation_table
        if table is None:
            masked_tokens = src_tokens.masked_fill(pred_mask, self.mask_id)
            return masked_tokens, src_tokens
        
        gt_tokens = table.to(src_tokens)[src_tokens]
        masked_tokens = gt_tokens.masked_fill(pred_mask, self.mask_id)
        _bos_id = self.alphabet.cls_idx if self.alphabet.prepend_bos else None
        _eos_id = self.eos_id if self.alphabet.append_eos else None
        masked_tokens = _pad_ends(
            masked_tokens, src_lengths, bos_value=_bos_id, eos_value=_eos_id, pad_value=self.pad_id
        )
        gt_tokens = _pad_ends(gt_tokens, src_lengths, bos_value=_bos_id, eos_value=_eos_id, pad_value=self.pad_id)
        return masked_tokens, gt_tokens


def _trim_ends(node: torch.Tensor, trim_bos=False, trim_eos=False) -> torch.Tensor:
    if trim_bos:
        node = node[:, 1:]
    if trim_eos:
        node = node[:, :-1]
    return node


def _pad_ends(
    node: torch.Tensor,
    eos_position: torch.LongTensor = None,
    bos_value: int | float | None = None,
    eos_value: int | float | None = None,
    pad_value: int | float | None = None,
) -> torch.Tensor:
    """
    Args:
        eos_position: position of the eos token (after padding both bos and eos tokens, if applicable)
    """
    assert node.ndim >= 2
    _pad_shape = list(node.shape)
    _pad_shape[1] = 1
    if bos_value is not None:
        node = torch.cat([node.new_full(_pad_shape, bos_value), node], dim=1)
    if eos_value is not None:
        assert pad_value is not None, "pad_value must be provided if eos_value is not None"
        assert eos_position is not None, "eos_position must be provided if eos_value is not None"
        node = torch.cat([node, node.new_full(_pad_shape, pad_value)], dim=1)
        for i in range(len(node)):
            node[i, eos_position[i]] = eos_value
    return node


def _split_chains(node: torch.Tensor, chain_index, mask):
    index = torch.zeros_like(chain_index)
    chain_nodes = []
    for b, _chain_index in enumerate(chain_index.unbind(0)):
        for _chain_id in torch.unique(_chain_index):
            _chain_mask = (_chain_index == _chain_id) * mask[b]
            if _chain_mask.sum() == 0:
                continue
            chain_nodes.append(node[b][_chain_mask])
            index[b].masked_fill_(_chain_mask, len(chain_nodes))
    return chain_nodes, index


def _merge_chains(chain_nodes: list[torch.Tensor], pad_value):
    lengths = torch.tensor([len(t) for t in chain_nodes]).to(device=chain_nodes[0].device)
    max_len = lengths.max()

    def _pad(t):
        seq_len = len(t)
        if seq_len >= max_len:
            return t[:max_len]
        padded_t = t.new_full((max_len - seq_len,), pad_value)
        return torch.cat([t, padded_t], dim=0)

    node = torch.stack([_pad(t) for t in chain_nodes], dim=0)
    return node, lengths


def _gather_chains(node: torch.Tensor, chain_lengths, index: torch.LongTensor):
    "Gathe chains after trimming special bos/eos tokens"
    num_chains = node.shape[0]
    assert node.ndim == 3 or node.ndim == 2
    gathered = node.new_zeros(index.shape + ((node.shape[-1],) if node.ndim == 3 else ()))
    for i in range(1, num_chains + 1):
        gathered[index == i] = node[i - 1][: chain_lengths[i - 1]]
    return gathered


def _find_lengths(
    node: torch.Tensor, pad_id: int, eos_id: int, use_eos: bool = False
) -> torch.LongTensor:
    lengths = torch.zeros(node.shape[0], dtype=torch.long)
    for i in range(node.shape[0]):
        if not use_eos:
            lengths[i] = (node[i] != pad_id).sum()
        else:
            lengths[i] = (node[i] == eos_id).nonzero()[0][0]
    return lengths
