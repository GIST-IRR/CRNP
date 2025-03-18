from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.fn import (
    stripe,
    diagonal_copy_,
    checkpoint,
)
import torch


class PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        viterbi=False,
        mbr=False,
        dropout=0.0,
        label=False,
    ):
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)

        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        # span_indicator = rule.new_ones(batch, N, N, NT).requires_grad_(
        #     viterbi or mbr
        # )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)
                # orig_sum = x.logsumexp(dim, keepdims=True)
                # prop = (x/2).log_softmax(dim)
                # return (prop + orig_sum).logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1)
                + Z[:, :, 1:-1, :].unsqueeze(-2),
                dim=2,
            ).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)
        # logZ = (s[torch.arange(batch), 0, lens] * root).sum(-1)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ,
                span_indicator,
                lens,
                tag_indicator=tag_indicator,
                mbr=mbr,
                label=label,
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_topk(self, rules, lens, viterbi=False, mbr=False, topk=1):
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)

        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, topk=None, dim=-1):
            if topk:
                dim_size = x.shape[dim]
                if dim_size < topk:
                    return x.topk(dim_size, dim)[0]
                else:
                    return x.topk(topk, dim)[0]
            else:
                return x.max(dim)[0]

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            b_n_x = b_n_x.logsumexp(-1)
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            l = Y.shape[2]
            b_n_yz = (Y[:, :, 1:-1, :, None] + Z[:, :, 1:-1, None, :]).reshape(
                batch, n, l - 2, -1
            )
            # b_n_x: b, n, l-2, None, NT*NT
            # rule: b, None, None, NT, NT*NT
            b_n_x = b_n_yz[:, :, :, None, :] + rule[:, None, None, :, :]
            b_n_x = b_n_x.permute(0, 1, 3, 2, 4)
            b_n_x = contract(b_n_x.reshape(batch, n, NT, -1), topk)
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1), topk)
            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
                continue

            x = terms.new_zeros(3, batch, n, NT, topk).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))
            x = x.permute(1, 2, 3, 0, 4).reshape(batch, n, NT, -1)

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, topk).logsumexp(-1)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root, topk)
        logZ = logZ.logsumexp(-1)

        if viterbi or mbr:
            # prediction = self._get_prediction(
            #     logZ, span_indicator, lens, tag_indicator, mbr=mbr
            # )
            # return {"partition": logZ, "prediction": prediction}
            return {"partition": logZ}
        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one(
        self,
        rules,
        lens,
        tree=None,
        viterbi=False,
        mbr=False,
        label=None,
        dropout=0.0,
    ):
        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)

        if tree is not None:
            if tree.shape[-1] == S:
                pos_tree = tree[..., NT:]
                tree = tree[..., :NT]

        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(
                Y[:, :, 1:-1, :].unsqueeze(-1)
                + Z[:, :, 1:-1, :].unsqueeze(-2),
                dim=2,
            ).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))

            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            return b_n_x

        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))

            return b_n_x

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            if tree is not None:
                child_mask = (
                    tree[:, torch.arange(n), torch.arange(n) + w] + 1e-9
                )
                child_mask = child_mask.log()
                if child_mask.dim() == 2:
                    child_mask = child_mask[..., None]

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]

            if w == 2:
                if tree is not None:
                    diagonal_copy_(
                        s,
                        (Xyz(Y_term, Z_term, X_y_z) + child_mask)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ],
                        w,
                    )
                else:
                    diagonal_copy_(
                        s,
                        Xyz(Y_term, Z_term, X_y_z)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ],
                        w,
                    )
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            if tree is not None:
                diagonal_copy_(
                    s,
                    (contract(x, dim=0) + child_mask)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )
            else:
                diagonal_copy_(
                    s,
                    contract(x, dim=0)
                    + span_indicator[:, torch.arange(n), torch.arange(n) + w],
                    w,
                )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, tag_indicator, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}
            # return {"partition": logZ}
        else:
            return {"partition": logZ}


class Faster_PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(
        self,
        rules,
        lens,
        dropout=None,
        tree=None,
        viterbi=False,
        mbr=False,
        label=False,
    ):
        assert viterbi == False

        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        # span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)
        span_indicator = rule.new_zeros(batch, N, N, NT).requires_grad_(
            viterbi or mbr
        )
        tag_indicator = rule.new_zeros(batch, N - 1, T).requires_grad_(
            viterbi or mbr
        )

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        terms = terms + tag_indicator  # to indicate viterbi tag

        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                # diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                diagonal_copy_(
                    s,
                    Xyz(Y_term, Z_term, X_y_z)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w, :
                    ],
                    w,
                )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            # diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
            diagonal_copy_(
                s,
                contract(x, dim=0)
                + span_indicator[:, torch.arange(n), torch.arange(n) + w, :],
                w,
            )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ,
                span_indicator,
                lens,
                mbr=mbr,
                label=label,
                tag_indicator=tag_indicator,
            )
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}

    @torch.enable_grad()
    def _inside_one(
        self,
        rules,
        lens,
        dropout=None,
        tree=None,
        viterbi=False,
        mbr=False,
        **kwargs,
    ):
        assert viterbi == False

        terms = rules["unary"]
        rule = rules["rule"]
        root = rules["root"]

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(
            viterbi or mbr
        )
        span_mask = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        if tree is not None:
            if tree.shape[-1] == S:
                pos_mask = tree[..., NT:]
                tree = tree[..., :NT]

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z, rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            x = torch.einsum("bny, bnz, bxyz -> bnx", y, z, rule)
            x = (
                (x + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bnwy, bnwz, bxyz -> bnwx", Y, Z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y - Y_normalizer.unsqueeze(-1)).exp(), (
                z - z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz->bnx", Y, z, rule)
            X = (
                (X + 1e-9).log()
                + Y_normalizer.unsqueeze(-1)
                + z_normalizer.unsqueeze(-1)
            )
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y - y_normalizer.unsqueeze(-1)).exp(), (
                Z - Z_normalizer.unsqueeze(-1)
            ).exp()
            X = torch.einsum("bny, bnz, bxyz-> bnx", y, Z, rule)
            X = (
                (X + 1e-9).log()
                + y_normalizer.unsqueeze(-1)
                + Z_normalizer.unsqueeze(-1)
            )
            return X

        for w in range(2, N):
            n = N - w

            # Get Mask
            if tree is not None:
                child_mask = (
                    tree[:, torch.arange(n), torch.arange(n) + w] + 1e-9
                )
                child_mask = child_mask.log()
                if child_mask.dim() == 2:
                    child_mask = child_mask[..., None]

            Y_term = terms[:, :n, :]
            Z_term = terms[:, w - 1 :, :]

            if w == 2:
                if tree is not None:
                    diagonal_copy_(
                        s,
                        (Xyz(Y_term, Z_term, X_y_z) + child_mask)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ].unsqueeze(-1),
                        w,
                    )
                else:
                    diagonal_copy_(
                        s,
                        Xyz(Y_term, Z_term, X_y_z)
                        + span_indicator[
                            :, torch.arange(n), torch.arange(n) + w
                        ].unsqueeze(-1),
                        w,
                    )
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            if tree is not None:
                diagonal_copy_(
                    s,
                    (contract(x, dim=0) + child_mask)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w
                    ].unsqueeze(-1),
                    w,
                )
            else:
                diagonal_copy_(
                    s,
                    contract(x, dim=0)
                    + span_indicator[
                        :, torch.arange(n), torch.arange(n) + w
                    ].unsqueeze(-1),
                    w,
                )

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(
                logZ, span_indicator, lens, mbr=mbr
            )
            return {"partition": logZ, "prediction": prediction}

        else:
            return {"partition": logZ}
