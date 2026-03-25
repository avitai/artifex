# Hyperparameter Search

`HyperparamSearchConfig` defines search strategy and the typed parameter
distribution objects used in the search space.

## Public Imports

```python
from artifex.configs import ChoiceDistribution, HyperparamSearchConfig, SearchType

config = HyperparamSearchConfig(
    name="baseline_search",
    search_type=SearchType.RANDOM,
    num_trials=20,
    search_space={
        "batch_size": ChoiceDistribution(
            param_path="training.batch_size",
            choices=(32, 64, 128),
        ),
    },
)
```

## Distribution Types

- `ParameterDistribution`
- `CategoricalDistribution`
- `UniformDistribution`
- `ChoiceDistribution`

The documented top-level convenience surface keeps the public examples on the
stable imports above. If you need a deeper distribution helper, import it from
its owning runtime module explicitly instead of assuming it exists on
`artifex.configs`.

## Search Types

- `SearchType.GRID`
- `SearchType.RANDOM`
- `SearchType.BAYESIAN`
- `SearchType.POPULATION`
