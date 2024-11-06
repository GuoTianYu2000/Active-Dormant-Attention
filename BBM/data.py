from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import sys
import pdb

from typing import List, Optional, Tuple

logging.getLogger().setLevel(logging.INFO)

def float_to_str(num, digits=3):
    if num>0:
        if digits==3:
            # Multiply by 100 and convert to integer to shift the decimal two places
            int_part = int(num * 100)
            # Format the integer to be zero-padded to 3 digits
            return f"{int_part:03d}"
        elif digits==4:
            # Multiply by 100 and convert to integer to shift the decimal two places
            int_part = int(num * 1000)
            # Format the integer to be zero-padded to 3 digits
            return f"{int_part:04d}"
    else:
        return "0"

@dataclass
class DataArgs:
    k: int = 5
    seq_length: int = 256
    fixed_special_toks: bool = True
    special_toks_offset: int = 0
    no_repeat: bool = False
    bos_num: int = 1
    delimiter_p: float = 0
    delim_num: int = 1
    mix_p: Optional[float] = None


class Dataset:
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        self.k = args.k
        self.seq_length = args.seq_length
        self.bos_num = args.bos_num
        self.train_test = train_test
        self.no_repeat = args.no_repeat
        self.delimiter_p = args.delimiter_p
        try:
            self.delim_num = args.delim_num
        except:
            self.delim_num = 1

        # init distributions
        self.meta = meta
        # self.meta = self.add_bos(self.meta)
        # self.meta = self.add_delimiter(self.meta)
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.num_tokens = self.meta['vocab_size']
        self.tok_range = list(np.arange(self.num_tokens))
        self.marginal = np.array(self.meta['marginal'])
        self.cond = np.array(self.meta['cond'])
        self.bos = self.meta['bos']
        self.delimiter = self.meta['delimiter']
        if "norm_tok_range" in self.meta.keys():
            self.norm_tok_range = self.meta['norm_tok_range']
        else:
            min_bos = np.min(self.bos)
            self.norm_tok_range = range(min_bos)
        if self.delim_num > 1:
            self.delimiter = self.meta['delimiter']

        # OOD
        if self.train_test is not None:
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # special tokens
        self.idxs = None
        if args.fixed_special_toks:
            if self.delim_num == 2:
                self.k = 2
                self.idxs = self.delimiter
            elif self.k>=0:
                # use unigram marginals
                self.idxs = list(np.array(self.marginal).argsort()[self.num_tokens-args.special_toks_offset-self.k:self.num_tokens-args.special_toks_offset])      
            else:
                self.idxs = [self.delimiter]

    def decode(self, idxs: List[int]) -> str:
        text = [self.itos[idx] for idx in idxs]
        return text
    
    def update_decoder(self, ):
        self.itos[-1] = ''
        self.itos[0] = '\\n'
        for i in self.idxs:
            self.itos[i] = 't'
    
    def update_cond(self, probs, idxs, p):
        probs_onehot = np.array([1 if i in idxs else 0 for i in self.tok_range])
        probs = (1 - p) * probs + p * probs_onehot
        return probs

    def uniform_transition(self, rng, subset):
        return rng.choice(subset)
    
    def uniform_transition_no_unseen(self, x, rng, drop=[]):
        if x in self.idxs:
            subset = set(np.where(self.cond[x, :] > 0)[0].tolist()) - set(self.idxs) - set(drop)
        else:
            subset = set(np.where(self.cond[x, :] > 0)[0].tolist()) - set(drop)
        subset = list(subset)
        x_next = rng.choice(subset)
        return x_next

    def update_identity_context(self, x, contexts):
        contexts[x] = x
        return contexts

    def update_previous_context(self, x, xp, contexts):
        if x not in self.idxs:
            return contexts
        elif x in contexts.keys():
            return contexts
        else:
            contexts[x] = xp
            return contexts

    def refresh_context(self, x, x_next, contexts, copy_toks):
        if x not in contexts.keys():
            return contexts
        elif contexts[x] in copy_toks:
            return contexts
        elif x_next in copy_toks:
            contexts[x] = x_next
            return contexts
        else:
            return contexts

    # prepare the context for icl
    def make_icl_context(self, triggers, rng, cond):
        # pdb.set_trace()
        if self.no_repeat:  # prevent next token to be same as idx
            pools = [self.norm_tok_range.copy() for idx in triggers]
            for i, idx in enumerate(triggers):
                pools[i].remove(idx)
        else:
            pools = [self.norm_tok_range for idx in triggers]
        # outs = [rng.choice(self.tok_range) for idx in idxs]
        # outs = [rng.choice(pool, p=(cond[idx][pool] / cond[idx][pool].sum())) for pool, idx in zip(pools, triggers)]
        outs = [rng.choice(pool) for pool in pools]
        context = [(t, o) for t, o in zip(triggers, outs)]
        context = dict(context)
        return context
    
    # (rely on make_icl_context) detect whether x in contexts.keys(), if so, just output the memory
    def icl_transition(self, x, rng, contexts):
        if x in contexts.keys():
            return contexts[x]
        else:
            return None
    
    # standard markov transition
    def markov_transition(self, x, rng):
        if x is None:
            return None
        else:
            probs = self.cond[x]
            return rng.choice(self.tok_range, p=probs)

    # gives a random conditional distribution for each seq (better to change to for each x)
    def perturb_cond(self, cond, rng):
        new_cond = []
        for i in range(self.num_tokens):
            new_cond.append(rng.dirichlet(cond[i]))
        return new_cond

    # It generates next token with customed conditional probabilities
    def custom_markov(self, x, rng, cond):
        probs = cond[x]
        return rng.choice(self.tok_range, p=probs)

    # It generates next token with customed probabilities
    def custom_iid(self, x, rng, probs):
        return rng.choice(self.tok_range, p=probs)

    # transition according to marginal
    def iid_transition(self, x, rng, ):
        return rng.choice(self.tok_range, p=self.marginal,)

    # start and end are indices of the start and the end of the copy
    def copy(self, x, rng, seq, start=None, end=None):
        if start is None or end is None:
            return None
        return seq[start:(end+1)]

    # initiate the bos tokens
    def bos_init(self, ):
        seq = []
        for idx in range(self.num_tokens):
            if idx in self.bos:
                seq.append(idx)
        return seq
    
    # It deletes trigger tokens in marginal and re-normalize probs to get the transition probabilities
    def no_trigger_init(self, rng):
        probs = self.marginal.copy()
        probs[self.idxs] = 0
        probs = probs / probs.sum()
        return probs
    
    def rand_init_no_trigger(self, rng):
        probs = rng.dirichlet(np.ones_like(self.norm_tok_range)).tolist()
        probs_all = probs + [0] * (len(self.tok_range) - len(self.norm_tok_range))
        probs_all = np.array(probs_all)
        probs_all[self.idxs] = 0
        probs_all = probs_all / probs_all.sum()
        return probs_all
    

    # it causes problem since some tokens only predict the trigger tokens
    def make_no_trigger_cond(self, zero_out_idxs):
        cond = self.cond.copy()
        for i in range(self.num_tokens):
            cond[i][zero_out_idxs] = 0
            if cond[i].sum() == 0:
                continue
            cond[i] /= cond[i].sum()
        return cond
    
    # It makes new conditional probabilities to be uniform in a given subset of tokens
    def make_subset_cond(self, subset):
        cond = self.cond.copy()
        for i in range(self.num_tokens):
            cond[i] = 0
            if i in subset or i in self.bos:
                cond[i][subset] = 1/len(subset)
        return cond
    
    """These functions are the first mechanism to generate the change of context delimiter. It randomly forces a delimtier token in the sequences. Since it has contradictions with the copy previous token mechanism, we stop using it."""
    # It generates a permutation of the conditional probabilities for a given list of subset
    def permute_cond(self, rng, subset):
        cond = self.cond.copy()
        permute_subset = [idx for idx in subset[1:]] + [subset[0]]
        cond[subset] = self.cond[permute_subset]
        return cond

    # It uniformly samples from self.max_length and use it the location for the change of context delimiter
    def get_delimiter_pos(self, rng):
        start = 5
        end = round(self.seq_length * 2 / 3)
        return rng.choice(list(range(start, end)))
    
    # It generates the delimiter when the delimiter_pos is reached
    def delimiter_transition(self, x, rng, idx, delimiter_pos):
        if idx == delimiter_pos:
            return self.stoi['<d>']
        else:
            return None
        
    """The second way is to use a fixed probability to generate the delimiter. The complication is that we need to set the probability to 0 after the first delimiter is generated."""
    def permute(self, subset):
        permute_subset = [idx for idx in subset[1:]] + [subset[0]]
        return permute_subset
    
    def permute_cond_no_delim(self, rng, subset):
        cond = self.cond.copy()
        for i in self.tok_range:
            cond[i, self.delimiter] = 0
            cond[i] = cond[i] / cond[i].sum()
        permute_subset = self.permute(subset)
        for i, n in enumerate(subset):
            if i < len(subset)-1:
                shift = permute_subset[i]
                cond[[n, shift]] = cond[[shift, n]]
                cond[:, [n, shift]] = cond[:, [shift, n]]
        return cond
    
    def permute_no_trigger_init(self, rng, subset):
        probs = self.marginal.copy()
        probs[self.idxs] = 0
        probs = probs / probs.sum()
        probs[subset] = probs[self.permute(subset)]
        return probs
        

    # It generates an OOD sequence without trigger tokens
    def no_trigger_gen_seq(self, rng: np.random.Generator, subset):
        seq = self.bos_init()
        cond = self.make_subset_cond(subset)
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov = self.custom_markov(x, rng, cond)
            if x in self.idxs:
                continue
            else:
                seq.append(x_markov)
        return seq


    # It is the default dgp in Biette's. All subclasses only need to rewrite gen_seq
    def gen_seq(self, rng: np.random.Generator):
        cond = None
        # cond = np.array([[1 for _ in range(self.num_tokens)] for _ in range(self.num_tokens)])
        contexts = self.make_icl_context(self.idxs, rng, cond)
        seq = self.bos_init()
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            if x_icl is None:
                seq.append(x_markov)
            else:
                seq.append(x_icl)

        return seq

    def gen_seqs(self, rng: np.random.Generator):
        while True:
            seq = self.gen_seq(rng)
            yield seq

    def gen_batch(self, rng: np.random.Generator, batch_size: int, ):
        seqs = []
        for _ in range(batch_size):
            seq = self.gen_seq(rng)
            seqs += seq
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        return x
    
    def gen_batch_ood(self, rng: np.random.Generator, batch_size: int, drop=[]):
        seqs = []
        for _ in range(batch_size):
            seq = self.gen_ood(rng, drop)
            seqs += seq
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        return x

    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos
def iterate_batches(dataset: Dataset,
                    batch_size: int = 20,
                    num_workers: int = 60,
                    seed: int = 42):
    def worker(queue, rng):
        while True:
            x = dataset.gen_batch(rng, batch_size)
            queue.put((x))

    import multiprocessing as mp
    q = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(q, np.random.default_rng([seed, i]))) for i in range(num_workers)]
    for p in processes:
        p.start()

    seq = []
    outputs_seq = []
    count = 0
    try:
        while True:
            x = q.get()
            yield (x[:,:-1], x[:,1:])
    except:
        for p in processes:
            p.kill()

class MetaProcess:
    def __init__(self, args: DataArgs,
                 train_test: Optional[str] = None,):
        self.k = args.k
        self.seq_length = args.seq_length
        self.bos_num = args.bos_num
        self.train_test = train_test
        self.no_repeat = args.no_repeat
        self.delimiter_p = args.delimiter_p
        self.args = args
        self.delim_num = args.delim_num

        # init distributions
        self.meta = pickle.load(open('data/meta.pkl', 'rb'))
        self.meta = self.add_bos(self.meta)
        if self.delim_num == 1:
            self.meta = self.add_delimiter(self.meta)
        else:
            self.meta, self.delim = self.add_double_delimiter(self.meta)
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.num_tokens = self.meta['vocab_size']
        
        self.tok_range = list(np.arange(self.num_tokens))

        # OOD
        if self.train_test is not None:
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # marginal probabilities over characters
        self.marginal = np.zeros(self.num_tokens)
        for k, cnt in self.meta['unigrams'].items():
            self.marginal[self.stoi[k]] = cnt
        self.marginal /= self.marginal.sum()
        self.marginal = self.marginal.tolist()

        # conditionals
        self.cond = [np.zeros(self.num_tokens) for _ in range(self.num_tokens)]
        for (w1, w2), cnt in self.meta['bigrams'].items():
            self.cond[self.stoi[w1]][self.stoi[w2]] += cnt
        for i in range(self.num_tokens):
            self.cond[i] /= self.cond[i].sum()
            self.cond[i] = self.cond[i].tolist()

    def update_cond(self, probs, idxs, p):
        p_t = 1 / len(idxs)
        probs_onehot = np.array([p_t if i in idxs else 0 for i in self.tok_range])
        probs = (1 - p) * probs + p * probs_onehot
        return probs
    
    def process(self,):
        seq = []
        bos_token = [f'<s_{i}>' for i in range(self.bos_num)]
        delim_token = []
        for idx in range(self.num_tokens):
            if self.itos[idx] in bos_token:
                seq.append(idx)
        mini_bos = np.min(seq)
        norm_tok_range = range(mini_bos)
        return {"marginal": self.marginal, "cond": self.cond, "itos": self.itos, "stoi": self.stoi, "vocab_size": self.num_tokens, "bos_num": self.bos_num, "delimiter_p": self.delimiter_p, "bos": seq, "delimiter": self.stoi['<d>'], "norm_tok_range": norm_tok_range}
    
    def two_delim_process(self,):
        seq = []
        bos_token = [f'<s_{i}>' for i in range(self.bos_num)]
        delim_token = []
        for idx in range(self.num_tokens):
            if self.itos[idx] in bos_token:
                seq.append(idx)
        mini_bos = np.min(seq)
        norm_tok_range = range(mini_bos)
        return {"marginal": self.marginal, "cond": self.cond, "itos": self.itos, "stoi": self.stoi, "vocab_size": self.num_tokens, "bos_num": self.bos_num, "delimiter_p": self.delimiter_p, "bos": seq, "delimiter": self.delim, "delim_num": self.delim_num, "norm_tok_range": norm_tok_range}
    
    def tuning(self, idxs, cutoff):
        for i in self.tok_range:
            if np.sum([self.cond[i, t] for t in self.idxs]) < cutoff:
                self.cond[i, :] = self.update_cond(self.cond[i, :], idxs[:1], cutoff)
        return [(t, np.sum([self.cond[t, t0] for t0 in idxs])) for t in self.tok_range]
        
    
    # here to make sure that <s> generates a new token following unigrams, and nothing generates <s>, nor <s> gets included in unigrams
    def add_bos(self, meta):
        bos = []
        for i in range(self.bos_num):
            idx = meta['vocab_size']
            tok = f'<s_{i}>'
            ref_pre = [(tok, 0) for tok in meta['unigrams'].keys()]
            ref_pre = dict(ref_pre)
            ref_post = meta['unigrams']
            ref_post[tok], ref_pre[tok] = 0, 0
            meta = self.update_meta(meta, idx, tok, ref_pre=ref_pre, ref_post=ref_post)
        return meta
    
    # here to make sure that 1. <d> is not in unigrams; 2. <d> generates a new token following unigrams; 3. all other tokens (except for <s>) generate <d> with probability p
    def add_delimiter(self, meta):
        idx = meta['vocab_size']
        tok = '<d>'
        ref_pre = [(tok, 0) for tok in meta['unigrams'].keys()]
        ref_pre = dict(ref_pre)
        for (w1, w2), cnt in self.meta['bigrams'].items():
            if w1 in [f'<s_{i}>' for i in range(self.bos_num)]:
                continue
            ref_pre[w1] += cnt
        for (w1, cnt) in ref_pre.items():
            ref_pre[w1] = cnt * self.delimiter_p / (1- self.delimiter_p)
        ref_post = meta['unigrams']
        ref_post[tok], ref_pre[tok] = 0, 0
        meta = self.update_meta(meta, idx, tok, ref_pre=ref_pre, ref_post=ref_post)
        return meta
    
    def add_double_delimiter(self, meta):
        delim = []
        ref_post = meta['unigrams']
        ref_pre = [(tok, 0) for tok in meta['unigrams'].keys()]
        ref_pre = dict(ref_pre)
        for (w1, w2), cnt in self.meta['bigrams'].items():
            if w1 in [f'<s_{i}>' for i in range(self.bos_num)] or w1 in [f'<d_{i}>' for i in range(self.delim_num)]:
                continue
            ref_pre[w1] += cnt
        
        for i in range(self.delim_num):
            idx = meta['vocab_size']
            tok = f'<d_{i}>'
            ref_pre_tmp = {}
            for (w1, cnt) in ref_pre.items():
                ref_pre_tmp[w1] = cnt * self.delimiter_p / (1 - self.delim_num * self.delimiter_p)
            ref_pre_tmp[tok] = 0
            ref_post[tok], ref_pre[tok] = 0, 0
            meta = self.update_meta(meta, idx, tok, ref_pre=ref_pre_tmp, ref_post=ref_post)
            delim.append(idx)
        return meta, delim
    
    def update_meta(self, meta, idx, tok, ref_pre=None, ref_post=None):
        assert meta['vocab_size'] == idx
        meta['itos'][idx] = tok
        meta['stoi'][tok] = idx
        meta['vocab_size'] += 1
        meta['unigrams'][tok] = 0
        for tok2 in meta['unigrams'].keys():
            meta['bigrams'][(tok, tok2)] = ref_post[tok2]
            meta['bigrams'][(tok2, tok)] = ref_pre[tok2]
        return meta

class icl(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test)
        self.description = "ONLY use ICL. At each round, we detect whether the current token occurs before, if so, we predict the token itself"
        self.expect = "(induction head (dormant when there's no repeated token)): detect whether the currect token occurs before"
        # we do not want delimiter to to interfere the results
        assert self.delimiter_p == 0

    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        contexts = {}
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_icl = self.icl_transition(x, rng, contexts)
            if x_icl is None:
                # pdb.set_trace()
                x_iid = self.iid_transition(x, rng)
                contexts = self.update_identity_context(x, contexts)
                seq.append(x_iid)
            else:
                seq.append(x_icl)
        return seq
    
    def special_test(self, seqs):
        raise NotImplementedError
    
class markov(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "ONLY use markov transition"
        self.expect = "(None). No need for attention mechanism"
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov = self.markov_transition(x, rng)
            seq.append(x_markov)
        return seq
    def get_triggers_pos(self, seqs):
        triggers_pos = np.full_like(seqs, 0) == 1
        return triggers_pos
    

class dormant_markov(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, predict (i+1) with (i-1)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.iid_transition(None, rng))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_markovp = self.markov_transition(x, rng), self.markov_transition(xp, rng)
            if x in self.idxs:
                seq.append(x_markovp)
            else:
                seq.append(x_markov)
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

class bbm(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, predict (i+1) with copying (i-1)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
        self.marginal2 = self.no_trigger_init(None)


    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        while len(seq) <= self.seq_length:
            if len(seq) == 1:
                x_markov = self.markov_transition(seq[-1], rng)
                seq.append(x_markov)
                continue
            x, xp = seq[-1], seq[-2]
            x_markov = self.markov_transition(x, rng)
            if x in self.idxs:
                seq.append(xp)
            else:
                seq.append(x_markov)
        return seq
    
    def gen_ood(self, rng: np.random.Generator, drop=[]):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        while len(seq) <= self.seq_length:
            seq.append(self.uniform_transition_no_unseen(seq[-1], rng, drop))
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos
    
# p=0 <-> markov, p=1 <-> bbm
class bbm_interpolate(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, with probability p, predict (i+1) with copying (i-1), and with probability 1-p predict (i+1) with markov(i-1)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
        self.mix_p = args.mix_p
        self.marginal2 = self.no_trigger_init(None)
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov = self.markov_transition(x, rng)
            if x in self.idxs:
                if rng.random() < self.mix_p:
                    seq.append(xp)
                else:
                    seq.append(x_markov)
            else:
                seq.append(x_markov)
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

# p=0 <-> markov, p=1 <-> dormant_markov
class dormant_markov_interpolate(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, with probability p, predict (i+1) with markov(i-1), and with probability 1-p predict (i+1) with markov(i)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
        self.mix_p = args.mix_p
        self.marginal2 = self.no_trigger_init(None)

    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_markovp = self.markov_transition(x, rng), self.markov_transition(xp, rng)
            if x in self.idxs:
                if rng.random() < self.mix_p:
                    seq.append(x_markovp)
                else:
                    seq.append(x_markov)
            else:
                seq.append(x_markov)
        return seq
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

class dormant_double_tasks(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p > 0
        self.description = "It is a mix of two heads, one with the same mechanism with bbm_2, and the other one is the change of context. Sepcifically, after the change of context delimiter, the all tokens except for triggers would get a fixed permutation."
        self.expect = "(L1: (H1: copy head, dormant when not on trigger tokens), (H2: delimiter detection head, dormant when there's no delimiter)))."
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        non_special_tok = [i for i in self.tok_range if i not in self.bos and i != self.delimiter]
        self.non_special_tok = non_special_tok
        self.cond2 = self.permute_cond_no_delim(None, non_special_tok)
        self.marginal2 = self.no_trigger_init(None)
        self.marginal3 = self.permute_no_trigger_init(None, non_special_tok)
        self.idxs2 = [(i - 1) % len(self.non_special_tok) for i in self.idxs]
    
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        delim_flag = False
        idxs = self.idxs
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            if x == self.delimiter:
                seq.append(self.custom_iid(None, rng, self.marginal3))
                delim_flag = True
                idxs = self.idxs2
                continue
            x_markov, x_markov2 = self.markov_transition(x, rng), self.custom_markov(x, rng, self.cond2)
            if delim_flag:
                if x in idxs:
                    seq.append(xp)
                else:
                    seq.append(x_markov2)
            else:
                if x in idxs:
                    seq.append(xp)
                else:
                    seq.append(x_markov)
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos


# This dgp may be incompatible with older dgps
class dormant_double_tasks_explore(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "test everything that makes double tasks work"
        self.expect = "(L1: (H1: copy head, dormant when on tokens that never generate triggers), (H2: delimiter detection head, dormant when there's no delimiter)))."
        # TODO: I need to make modification
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        non_special_tok = [i for i in self.tok_range if i not in self.bos and i != self.delimiter]
        self.non_special_tok = non_special_tok
        self.cond2 = self.permute_cond_no_delim(None, non_special_tok)
        self.marginal2 = self.no_trigger_init(None)
        self.marginal3 = self.permute_no_trigger_init(None, non_special_tok)
        self.idxs2 = [(i - 1) % len(self.non_special_tok) for i in self.idxs]
    
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        delim_pos = self.get_delimiter_pos(rng)
        delim_flag = False
        idxs = self.idxs
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            if x == self.delimiter:
                marginal2 = self.rand_init_no_trigger(rng)
                seq.append(self.custom_iid(None, rng, marginal2))
                delim_flag = True
                idxs = self.idxs2
                continue
            x_markov, x_markov2, x_delim = self.markov_transition(x, rng), self.custom_markov(x, rng, self.cond2), self.delimiter_transition(x, rng, len(seq), delim_pos)
            if x_delim is not None:
                seq.append(x_delim)
            elif delim_flag:
                if x in idxs:
                    seq.append(xp)
                else:
                    seq.append(x_markov2)
            else:
                if x in idxs:
                    seq.append(xp)
                else:
                    seq.append(x_markov)
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

# Try mixture of dormant markov and dormant copy
class dormant_double_tasks_explore1(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "test everything that makes double tasks work; Try mixture of dormant markov and dormant copy"
        self.expect = "(L1: (H1: copy head, dormant when on tokens that never generate triggers), (H2: delimiter detection head, dormant when there's no delimiter)))."
        # TODO: I need to make modification
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        non_special_tok = [i for i in self.tok_range if i not in self.bos and i != self.delimiter]
        self.non_special_tok = non_special_tok
        self.mix_p = args.mix_p
    
    def gen_seq(self, rng: np.random.Generator):
        p = rng.random()
        if p < self.mix_p:
            return self.gen_seq_markov(rng)
        else:
            return self.gen_seq_copy(rng)
    
    def gen_seq_markov(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_markovp = self.markov_transition(x, rng), self.markov_transition(xp, rng)
            if x in self.idxs:
                seq.append(x_markovp)
            else:
                seq.append(x_markov)
        return seq

    def gen_seq_copy(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov = self.markov_transition(x, rng)
            if x in self.idxs:
                seq.append(xp)
            else:
                seq.append(x_markov)
        return seq
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

# dormant Biette plus, add copy subgroup
class dormant_double_tasks_explore2(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "test everything that makes double tasks work"
        self.expect = "L1H1->L2H1"
        # TODO: I need to make modification
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        copy_toks = [i for i in self.norm_tok_range if i not in self.bos and i not in self.idxs and i < 50]
        self.copy_toks = copy_toks
    
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        # pdb.set_trace()
        contexts = self.make_icl_context(self.idxs, rng, None)
        occurance = dict([(idx, False) for idx in self.idxs])
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            if x in self.idxs:
                if not occurance[x]:
                    x_next = x_icl
                    occurance[x] = True
                elif contexts[x] in self.copy_toks:
                    x_next = x_icl
                else:
                    x_next = xp
            else:
                x_next = x_markov
            contexts = self.refresh_context(x, x_next, contexts, self.copy_toks)
            seq.append(x_next)
        return seq
    

    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

# dormant Biette plus the other version, add copy subgroup and lower the prob of having copy tokens
class dormant_double_tasks_explore4(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "comparing to explore2, now if copy tokens do not occur, we would simply use Markov transition"
        self.expect = "L1H1->L2H1"
        # TODO: I need to make modification
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        copy_toks = [i for i in self.norm_tok_range if i not in self.bos and i not in self.idxs and i < 40]
        self.copy_toks = copy_toks
    
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        # pdb.set_trace()
        contexts = self.make_icl_context(self.idxs, rng, None)
        occurance = dict([(idx, False) for idx in self.idxs])
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            if x in self.idxs:
                if not occurance[x]:
                    x_next = x_icl
                    occurance[x] = True
                elif contexts[x] in self.copy_toks:
                    x_next = x_icl
                else:
                    x_next = x_markov
            else:
                x_next = x_markov
            contexts = self.refresh_context(x, x_next, contexts, self.copy_toks)
            seq.append(x_next)
        return seq
    

    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

# two bos
class dormant_double_tasks_explore3(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        assert self.delimiter_p == 0
        self.description = "test everything that makes double tasks work"
        self.expect = "L1H1->L2H1"
        # TODO: I need to make modification
        assert self.bos_num == 2
        non_special_tok = [i for i in self.tok_range if i not in self.bos and i != self.delimiter]
        self.non_special_tok = non_special_tok
        self.cond2 = self.permute_cond_no_delim(None, non_special_tok)
        self.idxs2 = [(i - 1) % len(self.non_special_tok) for i in self.idxs]
        markov_tok = [i for i in self.norm_tok_range if i not in self.idxs]
        markov_tok1 = self.permute(markov_tok)
        self.tok_permute = [(markov_tok[i], markov_tok1[i]) for i in range(len(markov_tok))]
        self.tok_permute = dict(self.tok_permute)
        self.mix_p = args.mix_p

    
    def gen_seq(self, rng: np.random.Generator):
        # decide which task
        p = rng.random()
        if p < self.mix_p:
            seq = [self.bos[0]]
            use_permute = False
        else:
            seq = [self.bos[1]]
            use_permute = True
        
        # second token
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))

        # generate
        used_cond = self.cond2 if use_permute else self.cond
        used_idxs = self.idxs2 if use_permute else self.idxs
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov = self.custom_markov(x, rng, used_cond)
            if x in used_idxs:
                seq.append(xp)
            else:
                seq.append(x_markov)
        return seq
    

    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos




class dormant_two_kinds_copies(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "We want to use two kinds of copies"
        self.expect = "(L1: (H1: copy head1), (H2: copy head2)))."
        # TODO: I need to make modification
        # markov_tok = [i for i in self.tok_range if i not in self.idxs and i not in self.bos and i != self.delimiter]
        assert self.delim_num == 2
        self.mid_k = round(self.k / 2)
        self.idxs1 = self.idxs[:self.mid_k]
        self.idxs2 = self.idxs[self.mid_k:]
        self.marginal2 = self.no_trigger_init(None)
        markov_tok = [i for i in self.norm_tok_range if i not in self.idxs]
        markov_tok1 = self.permute(markov_tok)
        self.tok_permute = [(markov_tok[i], markov_tok1[i]) for i in range(len(markov_tok))]
        self.tok_permute = dict(self.tok_permute)
    
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.custom_iid(None, rng, self.marginal2))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov = self.markov_transition(x, rng)
            if x in self.idxs1:
                seq.append(xp)
            elif x in self.idxs2:
                seq.append(self.tok_permute[xp])
            else:
                seq.append(x_markov)
        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos


# I feel this dgp is not that necessary since it only adds a new procedure (L2) in bbm.
class dormant_Biette(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "Biette's setting with ICL becomes copying the previous token of the first occurance of the trigger instead of the following token. CAVEAT1: we cannot control the previous tokens of triggers, so we use the previous token of the first trigger, which may be a problem. CAVEAT2: we use rejection sampling to avoid getting triggers on the intial token."
        self.expect = "((L1: copy head, dormant when not on trigger tokens) -> L2: induction head, dormant when there's no repeated triggers)). When activated, the induction head will copy the information stored on the previous repeated trigger. Then use it to predict the next token."
        self.marginal2 = self.no_trigger_init(None)

    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        contexts = {}
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            contexts = self.update_previous_context(x, xp, contexts)
            if x_icl is not None:
                seq.append(x_icl)
            else:
                seq.append(x_markov)

        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos



class multitask(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "Two bos tokens, one for each task"
        self.expect = "two heads: one head taking responsibility for each task"
        self.marginal2 = self.no_trigger_init(None)
        

    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        marginal = self.rand_init_no_trigger(rng)
        seq.append(self.custom_iid(None, rng, marginal))
        contexts = {}
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            contexts = self.update_previous_context(x, xp, contexts)
            if x_icl is not None:
                seq.append(x_icl)
            else:
                seq.append(x_markov)

        return seq
    
    def get_triggers_pos(self, seqs):
        triggers_pos = np.isin(seqs, self.idxs)
        return triggers_pos

name_to_data = {'icl': icl, "markov": markov, "dormant_markov": dormant_markov, "bbm": bbm, "bbm_2": bbm, "dormant_double_tasks": dormant_double_tasks, "bbm_interpolate": bbm_interpolate, "dormant_markov_interpolate": dormant_markov_interpolate, "dormant_double_tasks_explore": dormant_double_tasks_explore, "dormant_double_tasks_explore1": dormant_double_tasks_explore1, "dormant_double_tasks_explore2": dormant_double_tasks_explore2, "dormant_double_tasks_explore3": dormant_double_tasks_explore3, "dormant_double_tasks_explore4": dormant_double_tasks_explore4, "dormant_two_kinds_copies": dormant_two_kinds_copies, "dormant_Biette": dormant_Biette, "default": Dataset}

def make_dataset(cfg, meta):
    # data_name is the orignal name
    return name_to_data[cfg.task_name](cfg.data_args, meta, train_test=None, )


def make_dataset_old(cfg, meta):
    # data_name is the orignal name
    return name_to_data[cfg.data_name](cfg.data_args, meta, train_test=None, )
