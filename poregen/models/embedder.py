import torch

from diffsci.models.nets import commonlayers


class PositionalEncoding1d(torch.nn.Module):
    def __init__(self, dembed, denominator=10000.0):
        super().__init__()
        self.dembed = dembed
        self.denominator = denominator
        indexes = torch.arange(start=0, end=dembed, step=2)  # [dembed//2]
        div_term = denominator ** (indexes / dembed)  # [dembed//2]
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        # x : [batch, seq_len]
        sin_cos = self.encode(x)
        return sin_cos

    def encode(self, x):
        # x : [batch, seq_len]
        sin = torch.sin(x.unsqueeze(-1)/self.div_term)  # [seq_len, dembed//2]
        cos = torch.cos(x.unsqueeze(-1)/self.div_term)  # [seq_len, dembed//2]
        sin_cos = (torch.stack([sin, cos], axis=-1).
                   flatten(start_dim=-2))  # [seq_len, dembed]
        return sin_cos

    def export_description(self):
        return {
            'dembed': self.dembed,
            'denominator': self.denominator
        }


class TwoPointCorrelationEmbedder(torch.nn.Module):
    def __init__(self,
                 dembed,
                 reduction: str | None = None,
                 scale: float = 30.0):
        super().__init__()
        self.dembed = dembed
        self.pos_encoder = PositionalEncoding1d(dembed)
        self.gaussian_proj = commonlayers.GaussianFourierProjection(dembed,
                                                                    scale)
        self.reduction = reduction
        self.scale = scale

    def forward(self, data):
        """
        Parameters
        ----------
        data: dict
            The input data dictionary, containing:
            dist : torch.Tensor of shape (..., seq_len)
                The distance between two points
            prob : torch.Tensor of shape (..., seq_len)
                The probability of the distance

        Returns
        -------
        x : torch.Tensor of shape (..., dembed) or (..., seq_len, dembed)
            The embedded tensor
        """
        dist = data['tpc_dist']
        prob = data['tpc_prob']
        x1 = self.pos_encoder(dist)
        x2 = self.gaussian_proj(-torch.log(prob+1e-6))
        x = x1 + x2
        if self.reduction == 'mean':
            x = x.mean(dim=-2)
        elif self.reduction is None:
            pass
        return x

    def export_description(self):
        return {
            'dembed': self.dembed,
            'reduction': self.reduction,
            'scale': self.scale
        }


class TwoPointCorrelationTransformer(torch.nn.Module):
    def __init__(self, tpc_embedder, nhead=4,
                 ffn_expansion=4, num_layers=2):
        super().__init__()
        self.embedder = tpc_embedder
        self.ffn_expansion = ffn_expansion
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embedder.dembed,
                nhead=nhead,
                dim_feedforward=self.embedder.dembed*ffn_expansion,
                batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        return x.mean(dim=1)

    def export_description(self):
        return {
            'embedder': self.embedder.export_description(),
            'encoder': {
                'd_model': self.embedder.dembed,
                'nhead': self.nhead,
                'ffn_expansion': self.ffn_expansion,
                'num_layers': self.num_layers
            }
        }


class PoreSizeDistEmbedder(torch.nn.Module):
    def __init__(self,
                 dembed,
                 reduction: str | None = None,
                 scale: float = 30.0,
                 type: str = 'psd_cdf'):
        super().__init__()
        self.dembed = dembed
        self.scale = scale
        self.pos_encoder = PositionalEncoding1d(dembed)
        self.gaussian_proj = commonlayers.GaussianFourierProjection(dembed,
                                                                    scale)
        self.reduction = reduction
        self.type = type

    def forward(self, data):
        """
        Parameters
        ----------
        data: dict
            The input data dictionary, containing:
            bins : torch.Tensor of shape (..., seq_len)
                The bins in the psd distribution
            density : torch.Tensor of shape (..., seq_len)
                The bins densities in the psd distribution

        Returns
        -------
        x : torch.Tensor of shape (..., dembed) or (..., seq_len, dembed)
            The embedded tensor
        """
        dist = data['psd_centers']
        density = data[self.type]
        x1 = self.pos_encoder(dist)
        x2 = self.gaussian_proj(density)
        x = x1 + x2
        if self.reduction == 'mean':
            x = x.mean(dim=-2)
        elif self.reduction is None:
            pass
        return x

    def export_description(self):
        return {
            'dembed': self.dembed,
            'reduction': self.reduction,
            'scale': self.scale
        }


class PoreSizeDistTransformer(torch.nn.Module):
    def __init__(self, psd_embedder, nhead=4,
                 ffn_expansion=4, num_layers=2):
        super().__init__()
        self.embedder = psd_embedder
        self.ffn_expansion = ffn_expansion
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embedder.dembed,
                nhead=nhead,
                dim_feedforward=self.embedder.dembed*ffn_expansion,
                batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        return x.mean(dim=1)

    def export_description(self):
        return {
            'embedder': self.embedder.export_description(),
            'encoder': {
                'd_model': self.embedder.dembed,
                'nhead': self.nhead,
                'ffn_expansion': self.ffn_expansion,
                'num_layers': self.num_layers
            }
        }


class PorosityEmbedder(torch.nn.Module):
    def __init__(self,
                 dembed,
                 scale=30.0):
        super().__init__()
        self.dembed = dembed
        self.scale = scale
        self.gaussian_proj = commonlayers.GaussianFourierProjection(
            dembed,
            scale
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dembed, 4*dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(4*dembed, 4*dembed),
            torch.nn.SiLU(),
            torch.nn.Linear(4*dembed, dembed)
        )

    def forward(self, x):
        # x : [nbatch, 1]
        x = x['porosity'].squeeze(-1)  # [nbatch]
        y = self.net(self.gaussian_proj(x))
        return y

    def export_description(self):
        return {
            'dembed': self.dembed,
            'scale': self.scale
        }


class PorosityVectorEmbedder(torch.nn.Module):
    def __init__(self, dembed, nhead=4,
                 ffn_expansion=4, num_layers=2, scale=30.0):
        super().__init__()
        self.dembed = dembed
        self.nhead = nhead
        self.ffn_expansion = ffn_expansion
        self.num_layers = num_layers
        self.scale = scale
        self.pos_encoder = PositionalEncoding1d(dembed)
        self.gaussian_proj = commonlayers.GaussianFourierProjection(dembed, scale)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.dembed,
                nhead=nhead,
                dim_feedforward=self.dembed*ffn_expansion,
                batch_first=True),
            num_layers=num_layers
        )

    def forward(self, data):
        """
        Parameters
        ----------
        data: dict
            The input data dictionary, containing:
            porosity : torch.Tensor of shape (..., seq_len)
                The vector of porosity values

        Returns
        -------
        x : torch.Tensor of shape (..., dembed)
            The embedded tensor
        """
        porosity_vector = data['porosity']  # [..., seq_len]
        seq_len = porosity_vector.shape[-1]
        positions = torch.arange(seq_len, device=porosity_vector.device)
        pos_encoding = self.pos_encoder(positions)  # [seq_len, dembed]

        porosity_encoding = self.gaussian_proj(porosity_vector)  # [..., seq_len, dembed]
        x = porosity_encoding + pos_encoding
        x = self.encoder(x)
        return x.mean(dim=1)

    def export_description(self):
        return {
            'dembed': self.dembed,
            'nhead': self.nhead,
            'ffn_expansion': self.ffn_expansion,
            'num_layers': self.num_layers,
            'scale': self.scale
        }


class PorosityVectorTransformer(torch.nn.Module):
    def __init__(self, porosity_vector_embedder, nhead=4,
                 ffn_expansion=4, num_layers=2):
        super().__init__()
        self.embedder = porosity_vector_embedder
        self.ffn_expansion = ffn_expansion
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embedder.dembed,
                nhead=nhead,
                dim_feedforward=self.embedder.dembed*ffn_expansion,
                batch_first=True),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        return x.mean(dim=1)

    def export_description(self):
        return {
            'embedder': self.embedder.export_description(),
            'encoder': {
                'd_model': self.embedder.dembed,
                'nhead': self.nhead,
                'ffn_expansion': self.ffn_expansion,
                'num_layers': self.num_layers
            }
        }


class MomentaEmbedder(torch.nn.Module):
    def __init__(self,
                 nmax,
                 dembed,
                 type: str = 'standardized_momenta',
                 scale=30.0,
                 fourier=True):
        super().__init__()
        self.dembed = dembed
        self.scale = scale
        if fourier:
            self.gaussian_proj = commonlayers.GaussianFourierProjectionVector(
                nmax,
                dembed,
                scale
            )
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dembed, 4*dembed),
                torch.nn.SiLU(),
                torch.nn.Linear(4*dembed, 4*dembed),
                torch.nn.SiLU(),
                torch.nn.Linear(4*dembed, dembed)
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(nmax, 4*dembed),
                torch.nn.SiLU(),
                torch.nn.Linear(4*dembed, 4*dembed),
                torch.nn.SiLU(),
                torch.nn.Linear(4*dembed, dembed)
            )
        self.fourier = fourier
        self.type = type

    def forward(self, data):
        # x : [nbatch, nmax]
        momenta = data[self.type]
        if self.fourier:
            y = self.net(self.gaussian_proj(momenta))
        else:
            y = self.net(momenta)
        return y

    def export_description(self):
        return {
            'dembed': self.dembed,
            'scale': self.scale
        }


class CompositeEmbedder(torch.nn.Module):
    def __init__(self, embedders):
        super().__init__()
        self.embedders = torch.nn.ModuleList(embedders)

    def forward(self, x):
        return torch.sum(
            torch.stack([embedder(x) for embedder in self.embedders],
                        dim=0), dim=0
        )

    def export_description(self):
        d = {}
        for i, embedder in enumerate(self.embedders):
            d[f'embedder_{i}'] = embedder.export_description()
        return d


def get_porosity_embedder(dembed, scale=30.0):
    return PorosityEmbedder(dembed, scale)


def get_psd_momenta_embedder(dembed, nmax=4, type='standardized_momenta', scale=30.0, fourier=True):
    return MomentaEmbedder(nmax, dembed, type=type, scale=scale, fourier=fourier)


def get_tpc_transformer(dembed,
                        nhead=4,
                        ffn_expansion=4,
                        num_layers=2,
                        scale: float = 30.0):
    embedder = TwoPointCorrelationEmbedder(dembed, scale=scale)
    transformer = TwoPointCorrelationTransformer(embedder,
                                                 nhead=nhead,
                                                 ffn_expansion=ffn_expansion,
                                                 num_layers=num_layers)
    return transformer


def get_psd_transformer(dembed,
                        type: str = 'psd_cdf',
                        nhead=4,
                        ffn_expansion=4,
                        num_layers=2,
                        scale: float = 30.0):
    embedder = PoreSizeDistEmbedder(dembed, scale=scale, type=type)
    transformer = PoreSizeDistTransformer(embedder,
                                          nhead=nhead,
                                          ffn_expansion=ffn_expansion,
                                          num_layers=num_layers)
    return transformer


def get_porosity_vector_transformer(dembed, nhead=4, ffn_expansion=4, num_layers=2, scale=30.0):
    embedder = PorosityVectorEmbedder(dembed, scale=scale)
    transformer = PorosityVectorTransformer(embedder, nhead=nhead, ffn_expansion=ffn_expansion, num_layers=num_layers)
    return transformer
