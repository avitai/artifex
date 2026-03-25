# Data Config

`DataConfig` is the typed dataclass for dataset selection and data-loading
behavior.

## Public Import

```python
from pathlib import Path

from artifex.configs import DataConfig

config = DataConfig(
    name="cifar10_data",
    dataset_name="cifar10",
    data_dir=Path("./data/cifar10"),
    split="train",
    num_workers=4,
    prefetch_factor=2,
)
```

## Key Fields

- `dataset_name`
- `data_dir`
- `split`
- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `shuffle`
- `drop_remainder`
- `prefetch_size`
- `augmentation`
- `augmentation_params`
- `validation_split`
- `test_split`
