from tqdm import tqdm # type: ignore
from pathlib import Path
from urllib.request import urlretrieve
from typing import *
import numpy as np
from functools import partial
import re
import torch
import mimetypes
import os
from fastprogress import master_bar, progress_bar

DATA_PATH=Path.home() / '.data'

################################################################################
# Downloading

class ProgressDownload(tqdm):
    def __init__(self, **kwargs):
        super().__init__(unit='B', unit_scale=True, unit_divisor=1024, **kwargs)

    def __call__(self, block, block_size, total_size):
        if total_size > 0:
            self.total = total_size
        # will also set self.n = block * block_size
        self.update(block * block_size - self.n)

    def __bool__(self):
        return not self.disable

def download(url, fname, force=False, progress_bar=True):
    fname = Path(fname)
    if force or not fname.exists():
        if fname.exists(): fname.unlink()
        fname.parent.mkdir(parents=True, exist_ok=True)
        with ProgressDownload(disable=not progress_bar, desc=str(fname)) as progress:
            urlretrieve(url, fname, reporthook=progress)


################################################################################
# Utility

def _list_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def list_files(path, recurse=False, extensions=None, include=None):
    path = Path(path)
    extensions = set(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
            res += _list_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _list_files(path, f, extensions)

def extensions_mime(ext):
    return frozenset(k for k,v in mimetypes.types_map.items() if v.startswith(ext))

EXTENSIONS_IMAGE = extensions_mime('image/')
EXTENSIONS_AUDIO = extensions_mime('audio/')
EXTENSIONS_VIDEO = extensions_mime('video/')

def img_files(path, recurse=False, extensions=None, include=None):
    if extensions is None: extensions = EXTENSIONS_IMAGE
    return list_files(path, recurse, extensions, include)

################################################################################
# Utility
def listify(o: Any) -> List:
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, (str, bytes)): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

# Recursive Cartesian product
def refract(f, x):
    if isinstance(f, Iterable):
        assert len(f) == len(x)
        return [refract(fi, xi) for fi, xi in zip(f, x)]
    elif isinstance(f, Callable):
        return f(x)
    # Could add dictionary, ...
    else:
        raise TypeError("Unkown type")

def comply(functions, x):
    if len(functions) > 0 and isinstance(functions[0], Iterable):
        assert len(functions) == len(x)
        return [comply(f, xi) for (f, xi) in zip(functions, x)]
    for f in functions:
        x = f(x)
    return x


def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

################################################################################
# ListContainer (for datasets)
################################################################################

# Can think of it as lazily applying tfms

class ListContainer():
    def __init__(self, items, tfms=None): self.items, self.tfms = list(items), list(tfms or [])

    def __getitem__(self, idx):
        if isinstance(idx, int): return comply(self.tfms, self.items[idx])
        elif isinstance(idx, slice): return self.__class__(self.items[idx], self.tfms)
        # Must be a list
        elif len(idx) > 0 and isinstance(idx[0], (bool, np.bool_)):
            if len(idx) != len(self.items):
                raise IndexError(f'Boolean index length {len(idx)} did not match collection length {len(self.items)}')
            assert len(idx) == len(self.items), "Boolean mask must have same length as object"
            return self.__class__([o for m,o in zip(idx, self.items) if m], self.tfms)
        else: return self.__class__([self.items[i] for i in idx], self.tfms)

    def exclude(self, idxs):
        if isinstance(idxs, slice): idxs = range(len(self))[idxs]
        if len(idxs) == 0: return self
        elif isinstance(idxs[0], (bool, np.bool_)):
            return self[[not x for x in idxs]]
        else:
            return self[[x for x in range(len(self)) if x not in idxs]]


    def split(self, by):
        return (self[by], self.exclude(by))

    def product(self, *others):
        for other in others:
            assert len(self) == len(other)
        lists = (self,) + others
        items = zip(*[getattr(l, 'items', l) for l in lists])
        tfms = [getattr(l, 'tfms') for l in lists]
        return self.__class__(items, tfms)

    def project_one(self, dim):
        return self.__class__([item[dim] for item in self.items], self.tfms[dim])

    def project(self):
        dim = len(self.tfms)
        return [self.project_one(i) for i in range(dim)]

    def __len__(self): return len(self.items)
    def __iter__(self): return (self[i] for i in range(len(self.items)))
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        if self.tfms: res += f'; Transformed by {self.tfms}'
        return res



################################################################################
## Optimizer
################################################################################

def maybe_update(os, dest, f):
    for o in os:
        for k,v in f(o).items():
            if k not in dest: dest[k] = v

def get_defaults(d): return getattr(d,'_defaults',{})

class Optimizer():
    def __init__(self, params, steppers, stats=None, **defaults):
        self.steppers = listify(steppers)
        # This ensures we load in the defaults into hyperparameters
        maybe_update(self.steppers, defaults, get_defaults)
        # Add statistics
        self.stats = listify(stats)
        # TODO: What? Why?
        maybe_update(self.stats, defaults, get_defaults)
        self.state = {}
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        for p,hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,hyper in self.grad_params():
            if p not in self.state:
                #Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats: state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p
sgd_step._defaults = dict(lr=1e-3)

def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p
weight_decay._defaults = dict(wd=0.)

opt_sgd = partial(Optimizer, steppers=[weight_decay, sgd_step])


################################################################################
## Learner for Training
################################################################################
_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order=0
    #def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name, caller):
        f = getattr(self, cb_name, None)
        if f and f(caller): return True
        return False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class TrainEvalCallback(Callback):
    def begin_fit(self, c):
        c.n_epochs=0.
        c.n_iter=0

    def after_batch(self, c):
        if not c.in_train: return
        c.n_epochs += 1./c.iters
        c.n_iter   += 1

    def begin_epoch(self, c):
        c.n_epochs=c.epoch
        c.model.train()

    def after_epoch(self, c):
        c.validate()

    def begin_validate(self, c):
        # This is a bit of a hack...
        self.training = c.model.training
        c.model.eval()

    def after_validate(self, c):
        if self.training:
            c.model.train()




def param_getter(m): return m.parameters()

class Learner():
    def __init__(self, model, data, loss_func, opt_func=opt_sgd, splitter=param_getter, cbs=None, cb_funcs=None):
        self.model,self.data,self.loss_func,self.opt_func,self.splitter = model,data,loss_func,opt_func,splitter
        self.in_train,self.logger,self.opt = False,print,None

        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)

    def add_cb(self, cb):
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)

    def fit_epoch(self):
        self.iters = len(self.dl)
        try:
            for i,(xb,yb) in enumerate(self.dl):
                try:
                    self.iter, self.xb, self.yb = i, xb, yb;        self('begin_batch')
                    self.pred = self.model(self.xb);                self('after_pred')
                    self.loss = self.loss_func(self.pred, self.yb); self('after_loss')
                    if not self.in_train: continue
                    self.loss.backward();                           self('after_backward')
                    self.opt.step();                                self('after_step')
                    self.opt.zero_grad()
                except CancelBatchException:                        self('after_cancel_batch')
                finally:                                            self('after_batch')
        except CancelEpochException:                                self('after_cancel_epoch')

    def validate(self):
        # Save the state so that it can be called from anywhere
        # If we passed these to fit_epoch (or similar) it wouldn't be necessary
        # But that means we'd need to pass it everywhere?
        dl = getattr(self, 'dl')
        in_train = getattr(self, 'in_train')

        with torch.no_grad():
            self.dl, self.in_train = self.data.valid_dl, False
            if not self('begin_validate'): self.fit_epoch()
        self('after_validate')
        self.dl, self.in_train = dl, in_train


    def fit(self, epochs, cbs=None, reset_opt=False, **defaults):
        self.add_cbs(cbs)
        if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), **defaults)

        try:
            self.epochs,self.loss = epochs,None; self('begin_fit')
            for epoch in range(epochs):
                # Dataloader creation on the fly???
                self.epoch,self.dl,self.in_train = epoch, self.data.train_dl, True
                if not self('begin_epoch'): self.fit_epoch()
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.remove_cbs(cbs)

    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_epoch', 'begin_validate', 'after_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name, self) and res
        return res

################################################################################
# Callbacks

class Progress(Callback):

    def begin_fit(self, c):
        self.mbar = master_bar(range(c.epochs))
        self.mbar.on_iter_begin()
        c.log = partial(self.mbar.write, table=True)
        c.plot = self.mbar.update_graph

    def after_fit(self, c): self.mbar.on_iter_end()
    def after_batch(self, c): self.pb.update(c.iter)
    def begin_epoch   (self, c): self.set_pb(c)
    def begin_validate(self, c): self.set_pb(c)

    def set_pb(self, c):
        self.pb = progress_bar(c.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(c.epoch)


class PlotLoss(Callback):

    def begin_fit(self, c):
        self.train_loss = []
        self.valid_loss = []

    def plot(self, c):
        c.plot([
            [np.arange(len(self.train_loss)) / c.iters, ewma(self.train_loss, 0.01)],
            [range(1, len(self.valid_loss)+1), self.valid_loss]
               ], [0., c.epochs])


    def after_loss(self, c):
        if c.in_train:
            self.train_loss.append(c.loss.item())
            if c.iter % 100 == 0:
                self.plot(c)

    def after_epoch(self, c):
        self.valid_loss.append(c.loss.item())
        self.plot(c)

def ewma(x, l=0.99):
    ans = x[:1]
    for _ in x[1:]:
        ans.append(ans[-1]*l + (1-l) * _)
    return ans


################################################################################
# Dataloader

# dataset
# batch_size, drop_last, sampler OR batch_sampler
# collate_fn
# timeout, worker_init_fn, pin_memory

# samplers:
# SequentialSampler(data)
# RandomSampler(data, repalcement=False, num_samples=None)
# WeightedRandomSampler(weights, num_samples, repalcement=True)
# DistributedSampler(dataset, num_replicas, rank)
