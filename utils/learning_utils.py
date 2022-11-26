def count_model_params(model):
    """
    Go through the model parameters
    """
    param_strings = []
    max_string_len = 126
    for name, param in model.named_parameters():
        if param.requires_grad:
            line = '.' * max(0, max_string_len - len(name) - len(str(param.size())))
            param_strings.append(f"{name} {line} {param.size()}")
    param_string = '\n'.join(param_strings)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_string, total_params, total_trainable_params


def _print_and_log(in_str, log_file):
    assert isinstance(in_str, str)
    print(in_str, flush=True)
    log_file.write(in_str + '\n')
    log_file.flush()
