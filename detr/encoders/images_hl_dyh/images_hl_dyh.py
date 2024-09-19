from typing import Dict, Union, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except Exception as e:
    pass


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def get_projector(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def get_high_res_encoder(
    model_name, pretrained=False, global_pool="", num_classes=0, local_weights_path=None
):
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        global_pool=global_pool,
        num_classes=num_classes,
    )
    if local_weights_path:
        model.load_state_dict(torch.load(local_weights_path), strict=False)
    return model


def get_mask_encoder(mask_in_chans, embed_dim, activation=nn.GELU):
    """
    Creates a mask encoder that downsamples a 3x224x224 RGB mask
    to an embed_dim-dimensional feature map of size 14x14.

    Args:
        mask_in_chans (int): Number of channels for the mask features after the first convolution.
        embed_dim (int): Desired output dimensionality for the feature map.
        activation (nn.Module): Activation function to use. Default is GELU.

    Returns:
        nn.Sequential: The mask encoder model.
    """
    return nn.Sequential(
        nn.Conv2d(
            3, mask_in_chans // 4, kernel_size=4, stride=4
        ),  # Reduce from 224x224 to 56x56
        LayerNorm2d(mask_in_chans // 4),
        activation(),
        nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        ),  # Reduce from 56x56 to 28x28
        LayerNorm2d(mask_in_chans),
        activation(),
        nn.Conv2d(
            mask_in_chans, embed_dim, kernel_size=2, stride=2
        ),  # Reduce from 28x28 to 14x14
        LayerNorm2d(embed_dim),
        activation(),
    )


def get_mask_encoder_dinov2(mask_in_chans, embed_dim, activation=nn.GELU):
    """
    Creates a mask encoder that downsamples a 3x224x224 RGB mask
    to an embed_dim-dimensional feature map of size 16x16.

    Args:
        mask_in_chans (int): Number of channels for the mask features after the first convolution.
        embed_dim (int): Desired output dimensionality for the feature map.
        activation (nn.Module): Activation function to use. Default is GELU.

    Returns:
        nn.Sequential: The mask encoder model.
    """
    return nn.Sequential(
        nn.Conv2d(
            3, mask_in_chans // 4, kernel_size=4, stride=4
        ),  # Reduce from 224x224 to 56x56
        LayerNorm2d(mask_in_chans // 4),
        activation(),
        nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        ),  # Reduce from 56x56 to 28x28
        LayerNorm2d(mask_in_chans),
        activation(),
        nn.Conv2d(
            mask_in_chans, embed_dim, kernel_size=2, stride=2
        ),  # Reduce from 28x28 to 14x14
        LayerNorm2d(embed_dim),
        activation(),
        nn.AdaptiveAvgPool2d((16, 16)),  # Adjust to 16x16
    )


def get_low_res_encoder(
    model_name, pretrained=False, global_pool="", num_classes=0, local_weights_path=None
):
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        global_pool=global_pool,
        num_classes=num_classes,
    )
    if local_weights_path:
        model.load_state_dict(torch.load(local_weights_path), strict=False)
    return model


def get_low_res_encoder_dinov2(
    model_name="vit_small_patch14_dinov2.lvd142m", local_weights_path=None
):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(local_weights_path)
    return model


def get_low_res_processor_dinov2(
    weights_path: str,
    input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
):
    model = get_low_res_encoder(
        model_name="vit_small_patch14_dinov2.lvd142m", local_weights_path=weights_path
    )
    data_config = timm.data.resolve_model_data_config(model)
    del model
    data_config["input_size"] = input_size
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x: N(HW)C
        x = x.permute(1, 0, 2)  # N(HW)C -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        # encoder to encode HR LR images
        high_res_encoder: Union[nn.Module, Dict[str, nn.Module]],
        low_res_encoder: Union[nn.Module, Dict[str, nn.Module]],
        # encoder to encoder rgb-mask
        mask_encoder: Union[nn.Module, Dict[str, nn.Module]],
        # to decide removed layers / aggregate features
        high_res_encoder_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        low_res_encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_high_res_encoder: bool = False,
        share_low_res_encoder: bool = False,
        share_mask_encoder: bool = False,
        # handle for nums of layers to remove when using conv
        downsample_ratio: int = 32,
        # to decide which models shold be frozen or not
        frozen: bool = False,
        pretrained: bool = False,
        # feature aggregation
        feature_aggregation: str = None,
        processor_model_weights_path: str = None,
    ):
        """
        Assumes high/low_res mask input: B,C,H,W
        """
        super().__init__()

        rgb_keys = list()
        mask_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = dict()
        key_shape_map = dict()

        if frozen:
            assert pretrained
            for param in high_res_encoder.parameters():
                param.requires_grad = False
            for param in low_res_encoder.parameters():
                param.requires_grad = False

        self.high_res_model_name = high_res_encoder_name
        self.low_res_model_name = low_res_encoder_name

        high_res_feature_dim = None
        low_res_feature_dim = None
        if high_res_encoder_name.startswith("convnext"):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(high_res_encoder.children())[:-2]
                high_res_encoder = torch.nn.Sequential(*modules)
                high_res_feature_dim = 768
            else:
                raise NotImplementedError(
                    f"Unsupported downsample_ratio: {downsample_ratio}"
                )

        if low_res_encoder_name.startswith("vit"):
            # siglip vit-B feature_dim=768
            low_res_feature_dim = 384
        else:
            raise NotImplementedError(
                f"Not using vit-B siglip, don't forget to change the feature dim"
            )

        if use_group_norm and not pretrained:
            high_res_encoder = replace_submodules(
                root_module=high_res_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )
            low_res_encoder = replace_submodules(
                root_module=low_res_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )
            mask_encoder = replace_submodules(
                root_module=mask_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )

        # handle sharing vision backbone
        if share_high_res_encoder:
            assert isinstance(high_res_encoder, nn.Module)
            key_model_map["high_res"] = high_res_encoder

        if share_low_res_encoder:
            assert isinstance(low_res_encoder, nn.Module)
            key_model_map["low_res"] = low_res_encoder

        if share_mask_encoder:
            assert isinstance(mask_encoder, nn.Module)
            key_model_map["mask"] = mask_encoder

        # config resizer and normalizer
        high_res_processor = get_low_res_processor_dinov2(
            processor_model_weights_path, input_size=(3, 512, 512)
        )
        low_res_processor = get_low_res_processor_dinov2(
            processor_model_weights_path, input_size=(3, 224, 224)
        )
        key_transform_map["high_res"] = high_res_processor
        key_transform_map["low_res"] = low_res_processor
        key_transform_map["mask"] = low_res_processor

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                if not share_high_res_encoder and not share_low_res_encoder:
                    print(
                        "both high and low res models maintain the best performance when pretrained"
                    )
            elif type == "mask":
                mask_keys.append(key)
                if not share_mask_encoder:
                    print(
                        "we only use mask in global vision, thus there will only be one model for mask"
                    )
            else:
                raise ValueError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        mask_keys = sorted(mask_keys)

        spacial_dim = len(rgb_keys) * 256 + len(mask_keys) * 256

        self.feature_aggregate_type = feature_aggregation
        if self.feature_aggregate_type == "attention_pool_2d":
            self.feature_aggregator = AttentionPool2d(
                spacial_dim=spacial_dim,
                embed_dim=low_res_feature_dim,
                num_heads=low_res_feature_dim // 64,
                output_dim=low_res_feature_dim,
            )
        elif self.feature_aggregate_type is None:
            self.feature_aggregator = nn.Identity()

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_high_res_model = share_high_res_encoder
        self.share_low_res_model = share_low_res_encoder
        self.share_mask_model = share_mask_encoder
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.key_shape_map = key_shape_map
        self.high_res_feature_dim = high_res_feature_dim
        self.low_res_feature_dim = low_res_feature_dim

        self.obs_uni_query_projector = nn.Sequential(
            nn.LayerNorm(self.low_res_feature_dim),
            nn.Linear(self.low_res_feature_dim, self.low_res_feature_dim),
        )
        self.obs_uni_key_projector = nn.Sequential(
            nn.LayerNorm(self.high_res_feature_dim),
            nn.Linear(self.high_res_feature_dim, self.low_res_feature_dim),
        )
        self.obs_uni_val_projector = nn.Sequential(
            nn.LayerNorm(self.high_res_feature_dim),
            nn.Linear(self.high_res_feature_dim, self.low_res_feature_dim),
        )

        out_put_dim = 512  # 384 raw

        self.img_projector = get_projector(
            input_dim=low_res_feature_dim, output_dim=out_put_dim
        )
        self.mask_projector = get_projector(
            input_dim=low_res_feature_dim, output_dim=out_put_dim
        )

    def aggregate_feature(self, feature):
        assert len(feature.shape) == 3
        return self.feature_aggregator(feature)

    def unified_resampler(self, low_res_raw_feature, high_res_raw_feature):
        # patchwise with square images
        # 除了siglip之外的其他vit结构需要去除cls token
        low_res_raw_feature = low_res_raw_feature[:, 1:]
        patch_num = int(low_res_raw_feature.shape[1] ** 0.5)
        # print(patch_num)
        patch_size = high_res_raw_feature.shape[-1] // patch_num
        # print(patch_size)
        # within patch attention
        high_res_raw_feature = high_res_raw_feature.permute(0, 2, 3, 1)
        high_res_raw_feature = high_res_raw_feature.reshape(
            len(high_res_raw_feature),
            patch_num,
            patch_size,
            patch_num,
            patch_size,
            high_res_raw_feature.shape[-1],
        )
        high_res_raw_feature = high_res_raw_feature.permute(0, 1, 3, 2, 4, 5)
        high_res_raw_feature = high_res_raw_feature.reshape(
            len(high_res_raw_feature),
            patch_num**2,
            patch_size**2,
            high_res_raw_feature.shape[-1],
        ).contiguous()

        # token attention
        embed_query = self.obs_uni_query_projector(low_res_raw_feature)
        embed_key = self.obs_uni_key_projector(high_res_raw_feature)
        embed_value = self.obs_uni_val_projector(high_res_raw_feature)
        embed_att = embed_query[:, :, None] @ (
            embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5)
        )
        embed_att = embed_att.nan_to_num()
        high_res_feature = (embed_att.softmax(-1) @ embed_value).mean(2)

        return low_res_raw_feature, high_res_feature

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process img for high/low res
        if self.share_high_res_model and self.share_low_res_model:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                # transform image
                high_res_img = self.key_transform_map["high_res"](img)
                # print("high_res_img.shape", high_res_img.shape)
                low_res_img = self.key_transform_map["low_res"](img)
                # print("low_res_img.shape", low_res_img.shape)
                # extract feature
                high_res_raw_feature = self.key_model_map["high_res"](high_res_img)
                # print("high_res_raw_feature.shape", high_res_raw_feature.shape)
                low_res_raw_feature = self.key_model_map["low_res"](
                    low_res_img
                ).last_hidden_state
                # print("low_res_raw_feature.shape", low_res_raw_feature.shape)
                # unify high/res features
                low_res_feature, high_res_feature = self.unified_resampler(
                    low_res_raw_feature=low_res_raw_feature,
                    high_res_raw_feature=high_res_raw_feature,
                )
                img_feature = low_res_feature + high_res_feature
                img_feature = self.img_projector(img_feature)
                # print("img_feature.shape", img_feature.shape)
                features.append(img_feature)
        else:
            print(
                "both high and low res models maintain the best performance when pretrained"
            )

        # process mask input
        for key in self.mask_keys:
            mask = obs_dict[key]
            if batch_size is None:
                batch_size = mask.shape[0]
            else:
                assert batch_size == mask.shape[0]
            assert mask.shape[1:] == self.key_shape_map[key]
            # transform mask
            mask = self.key_transform_map["mask"](mask)
            # print("mask.shape", mask.shape)
            mask_feature = self.key_model_map["mask"](mask)
            # print("mask_feature.shape", mask_feature.shape)
            mask_feature = mask_feature.flatten(start_dim=2).permute(0, 2, 1)
            mask_feature = self.mask_projector(mask_feature)
            features.append(mask_feature)

        # concatenate all features
        obs_feature = torch.cat(features, dim=1)
        # print("obs_raw_feature", obs_raw_feature.shape)
        if self.feature_aggregate_type is not None:
            obs_feature = self.aggregate_feature(obs_feature)
        # print("obs_feature.shape", obs_feature.shape)
        return obs_feature

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
