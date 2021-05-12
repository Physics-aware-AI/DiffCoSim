# modified from https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
# OOD to better manage memory usage

from graphviz import Digraph
import torch
from torch.autograd import Variable, Function


class BadGradFinder():
    def __init__(self):
        self.fn_dict = {}
        self.hooks = {}
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        self.dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    def delete(self):
        for fn in self.hooks:
            self.hooks[fn].remove()
        self.fn_dict.clear() # this is important!
        self.hooks.clear()
        del self.fn_dict
        del self.hooks
        del self.dot

    def iter_graph(self, root, callback):
        queue = [root]
        seen = set()
        while queue:
            fn = queue.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
            callback(fn)

    def hook_cb(self, fn):
        def register_grad(grad_input, grad_output):
            self.fn_dict[fn] = grad_input
        self.hooks[fn] = fn.register_hook(register_grad)

    def register_hooks(self, var):
        self.iter_graph(var.grad_fn, self.hook_cb)

    def is_nan_grad(self, grad_output):
        if grad_output is None:
            return False
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any()
    
    def build_graph(self, fn):
        if hasattr(fn, 'variable'):  # if GradAccumulator
            u = fn.variable
            node_name = 'Variable\n ' + self.size_to_str(u.size())
            self.dot.node(str(id(u)), node_name, fillcolor='lightblue')
        else:
            assert fn in self.fn_dict, fn
            if any(self.is_nan_grad(gi) for gi in self.fn_dict[fn]):
                fillcolor = 'red' ; grad = ''
            else:
                fillcolor = 'white'
                max_grad = 0 
                for gi in self.fn_dict[fn]:
                    if gi is None:
                        continue
                    if gi.nelement() == 0:
                        continue
                    max_grad = max(max_grad, abs(torch.max(gi).item()), abs(torch.min(gi).item()))
                if max_grad > 1e6:
                    fillcolor = 'red'
                grad = "({:.2e})".format(max_grad)

            self.dot.node(str(id(fn)), str(type(fn).__name__)+grad, fillcolor=fillcolor)
        for next_fn, _ in fn.next_functions:
            # print(id(next_fn), '->', id(fn))
            if next_fn is not None:
                next_id = id(getattr(next_fn, 'variable', next_fn))
                self.dot.edge(str(next_id), str(id(fn)))

    def size_to_str(self, size):
        return '('+(', ').join(map(str, size))+')'

    def make_dot(self, var):
        # print(f"len of fn_dict before build_buildgraph: {len(self.fn_dict)}")
        self.iter_graph(var.grad_fn, self.build_graph)



if __name__ == '__main__':
    x = Variable(torch.randn(10, 10), requires_grad=True)
    y = Variable(torch.randn(10, 10), requires_grad=True)

    z = x / (y * 0)
    z = z.sum() * 2
    hooks = BadGradFinder()
    hooks.register_hooks(z)

    z.backward()
    
    hooks.make_dot(z)
    hooks.dot.save('tmp.dot')
    hooks.delete()
    del hooks