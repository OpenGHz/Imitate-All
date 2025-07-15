import tyro
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil


class ConcatenateConfig(BaseModel):
    """Configuration for concatenating datasets.
    Args:
        datasets: A list of paths to the dataset directories to concatenate.
        output_path: The path to the output directory for the concatenated dataset.
    """

    datasets: List[Path]
    output_path: Path


def concatenate_datasets(config: ConcatenateConfig) -> None:
    """Concatenate multiple datasets into a single dataset.
    For example, consider there are two datasets:
    - dataset1
        - episode_1
        ...
        - episode_N
    - dataset2
        - episode_1
        ...
        - episode_N
    The Concatenated dataset will look like:
    - concatenated_dataset
        - episode_1
        - episode_2
        ...
        - episode_2N
    the order of the episodes is preserved in each dataset
    and the episodes are concatenated in the order they appear
    in the datasets list in the config.
    """
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    episode_cnt = 0
    for dataset in config.datasets:
        # sort the episodes in the dataset
        episodes = sorted(dataset.iterdir(), key=lambda x: x.stem)
        print(f"Sorted episodes in {dataset}: {episodes}")
        for episode in episodes:
            assert not episode.is_dir(), f"{episode} is a directory"
            # 直接进行重命名（而不是沿用之前的名称），从而统一了合并后的数据集的命名方式
            # 但后缀名仍然保留之前的，避免可能的格式问题
            new_episode_name = f"episode_{episode_cnt}" + episode.suffix
            print(f"Copying {episode} to {output_path / new_episode_name}")
            # Copy the episode to the output path with a new name
            shutil.copy(episode, output_path / new_episode_name)
            episode_cnt += 1


if __name__ == "__main__":
    config = tyro.cli(ConcatenateConfig)
    concatenate_datasets(config)
