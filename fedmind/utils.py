class StateDict(dict):
    """math operations enabled state dict for PyTorch Module
    ## Example
    >>> state_dict = StateDict({"a": 1, "b": 2})
    >>> state_dict + 1
    {'a': 2, 'b': 3}
    >>> state_dict + {"a": 1, "b": 2}
    {'a': 2, 'b': 4}
    >>> state_dict - 1
    {'a': 0, 'b': 1}
    >>> state_dict - {"a": 1, "b": 2}
    {'a': 0, 'b': 0}
    >>> -state_dict
    {'a': -1, 'b': -2}
    >>> state_dict * 2
    {'a': 2, 'b': 3}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = v + other[k]
        else:
            for k, v in self.items():
                res[k] = v + other
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = v - other[k]
        else:
            for k, v in self.items():
                res[k] = v - other
        return res

    def __rsub__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = other[k] - v
        else:
            for k, v in self.items():
                res[k] = other - v
        return res

    def __neg__(self):
        res = StateDict()
        for k, v in self.items():
            res[k] = -v
        return res

    def __mul__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = v * other[k]
        else:
            for k, v in self.items():
                res[k] = v * other
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = v / other[k]
        else:
            for k, v in self.items():
                res[k] = v / other
        return res

    def __rtruediv__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = other[k] / v
        else:
            for k, v in self.items():
                res[k] = other / v
        return res

    def __pow__(self, other):
        res = StateDict()
        if isinstance(other, dict):
            for k, v in self.items():
                res[k] = v ** other[k]
        else:
            for k, v in self.items():
                res[k] = v**other
        return res

    def add_(self, other):
        if isinstance(other, dict):
            for k in self.keys():
                self[k] += other[k]
        else:
            for k, v in self.items():
                self[k] += other

    def sub_(self, other):
        if isinstance(other, dict):
            for k in self.keys():
                self[k] -= other[k]
        else:
            for k in self.keys():
                self[k] -= other

    def mul_(self, other):
        if isinstance(other, dict):
            for k in self.keys():
                self[k] *= other[k]
        else:
            for k in self.keys():
                self[k] *= other

    def div_(self, other):
        if isinstance(other, dict):
            for k in self.keys():
                self[k] /= other[k]
        else:
            for k in self.keys():
                self[k] /= other

    def copy_(self, other):
        if isinstance(other, dict):
            for k in other.keys():
                self[k] = other[k]
        else:
            for k in self.keys():
                self[k] = other


class EasyDict(dict):
    """
    An dot accessible dictionary.

    Code modified from [easydict](https://github.com/makinacorpus/easydict) project.\\
    Licensed under the LGPL version 3.0.


    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1

    Bullet-proof

    >>> EasyDict({})
    {}
    >>> EasyDict(d={})
    {}
    >>> EasyDict(None)
    {}
    >>> d = {'a': 1}
    >>> EasyDict(**d)
    {'a': 1}
    >>> EasyDict((('a', 1), ('b', 2)))
    {'a': 1, 'b': 2}

    Set attributes

    >>> d = EasyDict()
    >>> d.foo = 3
    >>> d.foo
    3
    >>> d.bar = {'prop': 'value'}
    >>> d.bar.prop
    'value'
    >>> d
    {'foo': 3, 'bar': {'prop': 'value'}}
    >>> d.bar.prop = 'newer'
    >>> d.bar.prop
    'newer'


    Values extraction

    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> isinstance(d.bar, list)
    True
    >>> from operator import attrgetter
    >>> list(map(attrgetter('x'), d.bar))
    [1, 3]
    >>> list(map(attrgetter('y'), d.bar))
    [2, 4]
    >>> d = EasyDict()
    >>> list(d.keys())
    []
    >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
    >>> d.foo
    3
    >>> d.bar.x
    1

    Still like a dict though

    >>> o = EasyDict({'clean':True})
    >>> list(o.items())
    [('clean', True)]

    And like a class

    >>> class Flower(EasyDict):
    ...     power = 1
    ...     mean = {}
    ...     color = {"r": 100, "g": 0, "b": 0}
    ...
    >>> f = Flower()
    >>> f.power
    1
    >>> f.color.r
    100
    >>> f.mean.x = 10
    >>> f.mean.x
    10
    >>> f = Flower({'height': 12})
    >>> f.height
    12
    >>> f['power']
    1
    >>> sorted(f.keys())
    ['color', 'height', 'mean', 'power']

    update and pop items
    >>> d = EasyDict(a=1, b='2')
    >>> e = EasyDict(c=3.0, a=9.0)
    >>> d.update(e)
    >>> d.c
    3.0
    >>> d['c']
    3.0
    >>> d.get('c')
    3.0
    >>> d.update(a=4, b=4)
    >>> d.b
    4
    >>> d.pop('a')
    4
    >>> d.a
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'a'

    convert to pure dict
    >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
    >>> d.__class__.__name__
    'EasyDict'
    >>> p = d.to_dict()
    >>> p
    {'foo': 0, 'bar': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]}
    >>> p.__class__.__name__
    'dict'
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in (
                "update",
                "pop",
                "to_dict",
                "merge",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, EasyDict):
            value = EasyDict(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def merge(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            if isinstance(self.get(k), EasyDict) and isinstance(d[k], dict):
                self[k].merge(d[k])
            else:
                self[k] = d[k]

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)

    def to_dict(self):
        d = {}
        for name, value in self.items():
            if isinstance(value, (list, tuple)):
                d[name] = [x.to_dict() if isinstance(x, EasyDict) else x for x in value]
            else:
                d[name] = value.to_dict() if isinstance(value, EasyDict) else value
        return d


if __name__ == "__main__":
    import doctest

    doctest.testmod()
