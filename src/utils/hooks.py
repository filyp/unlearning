import torch as pt

from utils.training import trainable_modules


def save_act_hook(module, args):
    module.saved_act = args[0].detach().clone()


class CalcSimilarityHooks:
    def __init__(self, model, target_grad, control_grad=None):
        self.model = model
        self.target_grad = target_grad
        self.control_grad = control_grad
        self._hook_handles = []

    def __enter__(self):
        for n, p in self.model.named_parameters():
            p.param_name = n
        for l in self.model.model.layers:
            for module in [
                l.mlp.up_proj,
                l.mlp.gate_proj,
                l.mlp.down_proj,
                l.self_attn.q_proj,
                l.self_attn.k_proj,
                l.self_attn.v_proj,
                l.self_attn.o_proj,
            ]:
                h = module.register_full_backward_hook(self.calc_similarity_hook)
                self._hook_handles.append(h)
                h = module.register_forward_pre_hook(save_act_hook)
                self._hook_handles.append(h)

    def __exit__(self, type, value, traceback):
        for h in self._hook_handles:
            h.remove()

    def calc_similarity_hook(self, module, input_grad, output_grad):
        if module.weight.requires_grad is False:
            return

        # module.custom_grad = pt.einsum("bti,bto->oi", module.saved_act, output_grad[0])  # classic backprop
        custom_grad = pt.einsum("bti,bto->toi", module.saved_act, output_grad[0])
        module.weight.grad = None

        ref = self.target_grad[module.weight.param_name]

        module.weight.target_sim = []
        module.weight.self_sim = []
        for i in range(len(custom_grad)):
            pos = custom_grad[i]
            final = pos.detach().clone()
            if self.control_grad is not None:
                mask_ref = self.control_grad[module.weight.param_name]
                mask = mask_ref.sign() == pos.sign()
                final[mask] = 0

            module.weight.target_sim.append(float((ref * final).sum()))
            module.weight.self_sim.append(float((pos * final).sum()))


# #########################################################


def append_act_hook(module, args):
    last_act = args[0].detach().clone()
    last_act = last_act[0, 1:-1]
    module.saved_acts.append(last_act)


def append_grad_hook(module, args):
    last_grad = args[0].detach().clone()
    last_grad = last_grad[0, 1:-1]
    module.saved_grads.append(last_grad)


class CollectActAndGrad:
    def __init__(self, model):
        self.model = model
        self._hook_handles = []

    def __enter__(self):
        for n, module in trainable_modules(self.model):
            h = module.register_forward_pre_hook(append_act_hook)
            self._hook_handles.append(h)
            h = module.register_full_backward_pre_hook(append_grad_hook)
            self._hook_handles.append(h)
            module.saved_acts = []
            module.saved_grads = []

    def __exit__(self, type, value, traceback):
        for h in self._hook_handles:
            h.remove()
