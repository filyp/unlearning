import torch as pt


def save_act_hook(module, args):
    module.saved_act = args[0]


class CalcSimilarityHooks:
    def __init__(self, model, control_grad, target_grad):
        self.model = model
        self.control_grad = control_grad
        self.target_grad = target_grad
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
        # module.custom_grad = pt.einsum("bti,bto->oi", module.saved_act, output_grad[0])
        custom_grad = pt.einsum("bti,bto->toi", module.saved_act, output_grad[0])
        module.weight.grad = None  # to save memory

        # (ref.unsqueeze(0) * module.custom_grad).sum(dim=-1).sum(dim=-1)  # equivalent
        ref = self.control_grad[module.weight.param_name]
        module.weight.control_sim = [float((ref * pos).sum()) for pos in custom_grad]
        ref = self.target_grad[module.weight.param_name]
        module.weight.target_sim = [float((ref * pos).sum()) for pos in custom_grad]

        module.weight.self_sim = [float((pos * pos).sum()) for pos in custom_grad]
