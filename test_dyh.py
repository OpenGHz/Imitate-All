import hydra
from omegaconf import OmegaConf
import os
from detr.encoders.images_hl_dyh.images_hl_dyh import MultiImageObsEncoder


os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(
    config_path="detr/encoders/images_hl_dyh",
    config_name="default_config",
    version_base=None,
)
def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    encoder_hl_dyh: MultiImageObsEncoder = hydra.utils.instantiate(cfg.encoder)
    print(encoder_hl_dyh)


if __name__ == "__main__":
    main()
