# Artifex Example Documentation Design Guide

This guide defines how to design, write, maintain, and review Artifex examples.
It is specific to Artifex's current runtime and contributor contract:

- reader-facing tutorial examples are dual-format `.py` and `.ipynb` pairs
- verification and maintenance utilities under `examples/` may remain `.py` only
- repository workflows use `uv`
- examples target JAX and Flax NNX
- backend behavior follows JAX defaults instead of custom CUDA forcing
- contributor-facing example docs live under `docs/examples/`

## Purpose

Artifex examples must do three jobs at once:

1. Teach a concrete generative modeling concept.
2. Demonstrate the current supported Artifex API.
3. Run successfully as real code, not documentation theater.

An Artifex example is successful only when all three are true.

### Standalone Pedagogy Is Opt-In

A small number of shipped example triplets may remain as standalone pedagogy
rather than runtime-backed Artifex tutorials. Use this escape hatch only when a
concept is best taught as raw JAX/Flax NNX code and a truthful runtime-backed
Artifex tutorial does not yet exist.

If you publish a standalone walkthrough:

- label the docs page with a status line that says `Standalone pedagogy`
- say in both the script/notebook source and the docs page that it does not instantiate shipped Artifex runtime owners
- group it under a standalone section in `docs/examples/index.md`
- prefix its published nav label with `Standalone`
- do not claim that it uses the Artifex framework or point to nonexistent neighbor examples

### Exploratory Workflows Are Opt-In

Some published example triplets may remain exploratory rather than canonical
runtime-backed tutorials. Use this bucket when the workflow is still useful for
contributors or advanced readers but currently relies on lower-level Artifex
components, custom orchestration, or a narrower research path than the public
example catalog should otherwise imply.

If you publish an exploratory workflow:

- label the docs page with a status line that says `Exploratory workflow`
- say in both the script/notebook source and the docs page what lower-level owners it uses
- state explicitly when it does not instantiate the canonical top-level owner or facade its filename suggests
- group it under an exploratory section in `docs/examples/index.md`
- prefix its published nav label with `Exploratory`
- do not describe it as production-ready or as the canonical Artifex tutorial for that family

### Validation Utilities Are Not Canonical Tutorials

A published validation-oriented pair may exist when readers benefit from a quick
environment or technology-stack check, but that material is not the same thing
as a supported model tutorial. Validation content must stay explicit about its
limited scope.

If you publish a validation utility:

- label the docs page with a status line that says `Validation utility`
- say in both the script/notebook source and the docs page that it does not instantiate shipped Artifex runtime owners when that is true
- group it under a validation section in `docs/examples/index.md`
- prefix its published nav label with `Validation`
- avoid presenting environment checks or raw NNX probes as canonical modeling tutorials

## Scope

This guide applies to:

- runnable example sources under `examples/`
- paired example notebooks under `examples/`
- example documentation pages under `docs/examples/`
- example templates under `docs/examples/templates/`

It also informs contributor-facing documentation in:

- `CONTRIBUTING.md`
- `docs/community/contributing.md`
- `examples/README.md`

## Design Principles

### Teach Through Real Execution

Examples should run end to end with the commands shown in the docs. Prefer small
working pipelines over aspirational pseudo-workflows.

### Prefer Present-Tense API Guidance

Document the current supported Artifex surface. Avoid transitional language and
avoid teaching historical implementation details.

### Progressive Disclosure

Each example should start with the smallest useful path, then add complexity in
clear stages:

1. environment and prerequisites
2. imports and configuration
3. model or pipeline construction
4. execution
5. inspection of outputs
6. optional extensions or experiments

### CPU-Safe by Default

Examples should run on CPU unless the example is inherently GPU-bound. GPU use is
an optimization path, not an excuse for unclear setup.

### GPU Requirements Must Be Explicit

If an example requires GPU, say so at the top and provide a direct verification
step with:

```bash
source ./activate.sh
uv run python scripts/verify_gpu_setup.py --require-gpu
```

If GPU is optional, say that clearly and avoid device forcing logic inside the
example.

### Source of Truth Lives in the Example Pair

For dual-format tutorial examples, the runnable `.py` and `.ipynb` pair is the
technical source of truth. The corresponding `docs/examples/...` page explains
the example, but should not drift from the actual source files.

## Documentation Architecture

Every substantial reader-facing tutorial example should have three aligned artifacts:

1. `examples/.../example_name.py`
   The runnable Jupytext-backed Python source.
2. `examples/.../example_name.ipynb`
   The paired notebook generated from the Python source.
3. `docs/examples/.../example-name.md`
   The reader-facing documentation page.

Use this split intentionally:

- `.py` is the easiest source to review and refactor.
- `.ipynb` supports notebook-first exploration.
- `.md` explains context, expected outcomes, and navigation.

Utility scripts under `examples/` are different. If the file exists to verify,
benchmark, or maintain the example surface rather than teach a workflow, it can
stay as a Python-only script with no paired notebook.

## Location Strategy

Put examples where users will expect them:

- `examples/generative_models/basic/` for short foundational examples
- `examples/generative_models/advanced/` for longer multi-stage examples
- `examples/generative_models/protein/` for protein-specific workflows
- `examples/generative_models/geometric/` for geometric modeling
- `examples/generative_models/sampling/` for sampling workflows

Put documentation under the corresponding `docs/examples/` section so the docs
match the runnable source domain.

## Dual-Format Workflow

Reader-facing tutorial examples should be maintained as Jupytext pairs. Use the
repo tool, not ad hoc manual notebook edits.

### Create a New Example

Start from:

- `docs/examples/templates/example_template.py`
- `docs/examples/templates/example_template.ipynb`

### Sync the Pair

Use:

```bash
source ./activate.sh
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py
```

### Validate the Pair

Use:

```bash
source ./activate.sh
uv run python scripts/jupytext_converter.py validate examples/path/to/
```

The Python file should remain the main review surface. The notebook should be
regenerated from it rather than hand-edited independently.

Roadmap-only topics belong outside `docs/examples/` until a runnable pair
exists. If a concept is still relevant but unshipped, track it in
`docs/roadmap/planned-examples.md` or the appropriate conceptual guide instead
of publishing an md-only example page.

Do not create notebook pairs for verification or maintenance scripts such as
`examples/verify_examples.py`.

## Runtime and Backend Contract

### Activation

Always show:

```bash
source ./activate.sh
```

Keep the repository-relative `./` prefix when documenting activation.

### Execution

Run examples through `uv`:

```bash
uv run python examples/path/to/example.py
```

Do not document direct `python ...` commands as the primary path.

### Backend Selection

Let JAX select the best available backend by default. Do not teach or embed:

- hard-coded multi-platform fallback lists
- system CUDA toolkit paths
- custom CUDA library path management

When backend visibility matters, use the verifier script:

```bash
uv run python scripts/verify_gpu_setup.py --json
```

### Device Statements

At the top of each example page, state one of:

- `CPU-compatible`
- `GPU-optional`
- `GPU-required`

If GPU-required, include the verifier command. If CPU-compatible, say so
explicitly.

## Code Standards for Examples

### Use the Supported Runtime Surface

Examples should prefer public imports such as:

```python
from artifex.configs import TrainingConfig
from artifex.generative_models.factory import create_model
```

Avoid reaching through private or transitional paths when a supported top-level
surface exists.

### Use Flax NNX

Artifex examples should use Flax NNX patterns consistently:

- `nnx.Module`
- `nnx.Rngs`
- explicit RNG flow
- current optimizer and training APIs

### Use Typed Configs

Prefer the typed `artifex.configs` surface and the current frozen dataclass
configuration model.

### Keep Side Effects Explicit

Examples may print or log progress because they are educational artifacts, but
they should not mutate global backend state, modify tracked repo files, or depend
on hidden shell state.

### Avoid Fake Code

Do not include placeholder commands, nonexistent files, or invented outputs.
If a section is conceptual, label it as conceptual. If code is runnable, make it
actually runnable.

## Writing the Companion Docs Page

Every `docs/examples/...` page should include:

- what the example demonstrates
- runtime estimate
- device requirements
- prerequisites
- exact execution command
- a short map of the major sections
- a few key code excerpts
- expected outputs or result shape
- related examples

Reader-facing `docs/examples/...` pages are package-user documentation, not
contributor setup notes. Their quick-start commands should therefore use the
shipped entrypoint directly, such as `python ...` or `jupyter lab ...`, and
should not require `source ./activate.sh`, `uv sync`, or `uv run ...` on the
page itself. Keep the repo-maintainer `uv` workflow in contributor surfaces
such as `examples/README.md`, `CONTRIBUTING.md`, and the author workflow below.

Good top-of-page structure:

1. title
2. difficulty or duration
3. runtime and device note
4. overview
5. quick start
6. key concepts
7. important code excerpts
8. next steps

## Output Capture Requirements

Examples should make it easy for readers to recognize success.

Preferred techniques:

- show expected tensor shapes
- show a few key scalar outputs
- describe saved artifacts or plots
- include short expected-output snippets in docs when stable

Avoid long unstructured logs in docs pages.

## Visual and Narrative Style

Write example docs for comprehension, not marketing.

Use:

- concrete section titles
- short explanatory paragraphs
- code excerpts with commentary
- direct statements about tradeoffs and limitations

Avoid:

- vague hype
- unexplained jargon
- giant code dumps with no framing
- duplicated prose between the docs page and the example source

## Maintenance Rules

When an example changes, review all three surfaces:

1. example `.py`
2. paired `.ipynb`
3. `docs/examples/...` page

Also update contributor-facing surfaces if the workflow contract changed:

- `examples/README.md`
- `CONTRIBUTING.md`
- `docs/community/contributing.md`

## Review Checklist

Before merging an example:

- [ ] the example runs with `source ./activate.sh`
- [ ] the example runs with `uv run python ...`
- [ ] device requirements are stated accurately
- [ ] the `.ipynb` pair was regenerated with `scripts/jupytext_converter.py`
- [ ] the docs page matches the current source
- [ ] imports use the supported Artifex surface
- [ ] commands use `uv`
- [ ] notebook and docs do not teach hidden backend forcing
- [ ] the example teaches a concrete concept, not just an API dump

## Recommended Author Workflow

```bash
source ./activate.sh

# create or edit the Python source
$EDITOR examples/path/to/example.py

# regenerate the notebook pair
uv run python scripts/jupytext_converter.py sync examples/path/to/example.py

# run the example
uv run python examples/path/to/example.py

# update the docs page
$EDITOR docs/examples/path/to/example-name.md
```

If the example is GPU-required, add:

```bash
uv run python scripts/verify_gpu_setup.py --require-gpu
```

## Related Files

- `docs/examples/templates/example_template.py`
- `docs/examples/templates/example_template.ipynb`
- `examples/README.md`
- `examples/EXAMPLES_GUIDE.md`
- `scripts/jupytext_converter.py`
- `docs/community/contributing.md`
- `CONTRIBUTING.md`
