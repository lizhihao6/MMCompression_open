from .glow_modules import *


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed,
                 non_local):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        c_out = in_channels // 2 if flow_coupling == "additive" else in_channels
        if non_local:
            self.block = MaskBlock(in_channels // 2, c_out, hidden_channels)
        else:
            self.block = Block(in_channels // 2, c_out, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            # logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            # logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
            self,
            input_channels,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            non_local
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        self.K = K
        self.L = L

        C = input_channels

        for i in range(L):
            # 1. Squeeze
            C *= 4
            self.layers.append(SqueezeLayer(factor=2))

            # 2. K FlowStep
            for j in range(K):
                _non_local = True if non_local and i == L - 1 and j >= K - 2 else False
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        non_local=_non_local
                    )
                )

            # 3. Split2d
            # if i < L - 1:
            #     self.layers.append(Split2d(num_channels=C))
            #     C = C // 2

    def forward(self, input, reverse=False):
        if reverse:
            return self.decode(input)
        else:
            return self.encode(input)

    def encode(self, z, logdet=None):
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        assert logdet is None
        return z

    # def decode(self, z, temperature=None):
    #     for layer in reversed(self.layers):
    #         if isinstance(layer, Split2d):
    #             z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
    #         else:
    #             z, logdet = layer(z, logdet=0, reverse=True)
    #     return z

    def decode(self, z):
        for layer in reversed(self.layers):
            z, logdet = layer(z, logdet=None, reverse=True)
            assert logdet is None
        return z


class Glow(nn.Module):
    def __init__(
            self,
            input_channels,
            hidden_channels=128,
            K=8,
            L=4,
            actnorm_scale=1.0,
            flow_permutation="invconv",
            flow_coupling="affine",
            LU_decomposed=True,
            non_local=False
    ):
        super().__init__()
        self.flow = FlowNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            non_local=non_local
        )

    def forward(self, x, reverse=False):
        return self.flow(x, reverse=reverse)
