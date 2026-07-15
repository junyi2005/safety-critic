import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding


class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        # Imported lazily so that the rest of navdp_safety.models (in particular
        # the safety critic) can be used without a Depth-Anything-V2 checkout;
        # it is only needed to build the RGB-D encoders.
        from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # Caveat: preprocess_mean/std are plain tensor attributes rather than registered
        # buffers, so they are not moved by .to(device); they are moved explicitly in
        # forward() instead.

        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.eval()
        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)

    @staticmethod
    def _to_channels_first(x, channels):
        """Return x as [..., C, H, W], accepting either channels-first or channels-last.

        The dataset emits channels-first (e.g. RGB [B, 3, H, W]) while the
        deployment path feeds channels-last (e.g. [B, H, W, 3]). Earlier
        revisions permuted unconditionally, which silently transposed
        channels-first inputs into garbage instead of raising: RGB planes with
        means 0/1/2 came back as 0.996/1.000/1.004, and depth [B, 1, H, W]
        became [B, H, 1, W]. Dispatch on the channel axis instead.
        """
        if x.shape[-3] == channels:
            return x                                    # already [..., C, H, W]
        if x.shape[-1] == channels:
            return x.movedim(-1, -3)                    # [..., H, W, C] -> [..., C, H, W]
        raise ValueError(
            f"cannot locate a channel axis of size {channels} in tensor of shape "
            f"{tuple(x.shape)}; expected [..., {channels}, H, W] or [..., H, W, {channels}]"
        )

    def _prepare(self, x, channels):
        """Normalize layout to [N, C, image_size, image_size]; report (B, T)."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        has_time = x.dim() == 5
        x = self._to_channels_first(x, channels)
        B = x.shape[0]
        T = x.shape[1] if has_time else None
        x = x.reshape(-1, channels, self.image_size, self.image_size)
        return x, B, T

    def forward(self, images, depths):
        # The pretrained DINOv2 encoders stay frozen; everything after them
        # (fusion transformer, projection) must receive gradients, so only the
        # encoder calls are wrapped in no_grad. Wrapping the whole forward --
        # as earlier revisions did -- froze former_net and project_layer too,
        # leaving the RGB-D branch untrainable.
        img, B, T = self._prepare(images, 3)
        with torch.no_grad():
            mean = self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)
            std = self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
            image_token = self.rgb_model.get_intermediate_layers((img - mean) / std)[0]
        if T is not None:
            image_token = image_token.reshape(B, T * 256, -1)

        dep, Bd, Td = self._prepare(depths, 1)
        with torch.no_grad():
            dep = dep.expand(-1, 3, -1, -1)
            depth_token = self.depth_model.get_intermediate_layers(dep)[0]
        if Td is not None:
            depth_token = depth_token.reshape(Bd, Td * 256, -1)

        tokens = torch.concat((image_token, depth_token), dim=1)
        former_token = tokens + self.former_pe(tokens)
        former_query = self.former_query(
            torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=self.device)
        )
        memory_token = self.former_net(former_query, former_token)
        return self.project_layer(memory_token)
