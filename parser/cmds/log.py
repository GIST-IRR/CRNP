def log_weight_histogram(writer, model, step, tag="train"):
    for k, v in model.named_parameters():
        writer.add_histogram(f"{tag}/{k}", v, step)
        writer.add_histogram(f"{tag}/{k}.grad", v.grad, step)


def log_rule_prob(writer, model, step, tag="train", logit=True):
    for k, v in model.rules.items():
        if v.dim() > 2:
            v = v.reshape(model.NT, -1)

        def log(value, prefix="log"):
            writer.add_embedding(
                value, tag=f"train/{prefix}_{k}", global_step=step
            )
            writer.add_histogram(f"{tag}/hist_{prefix}_{k}", value, step)

        if logit or k in ["root", "unary"]:
            log(v, prefix="logit")
            v = v.exp()
            log(v, prefix="prob")
        else:
            log(v, prefix="prob")
            v = v.log()
            log(v, prefix="log")
