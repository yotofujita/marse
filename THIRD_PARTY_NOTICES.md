# Third-Party Notices

This repository contains the MARSE training, inference, and evaluation code. It does not vendor the generated LibriMix, LibriSpeech, WHAM!, DEMAND, or LibriDEMAND audio datasets.

## Git Submodules

The dataset generation repositories are included as Git submodules and keep their own licenses and citation requirements:

- `external/LibriMix`: https://github.com/JorisCos/LibriMix
- `external/Libri1MixDEMAND`: https://github.com/yotofujita/Libri1MixDEMAND

Review the license and data-use terms in each submodule before redistributing generated data or derived artifacts.

## Generated Data

The generated datasets depend on third-party source corpora:

- LibriSpeech
- WHAM! noise
- DEMAND noise

The generated audio is subject to the licenses and terms of those datasets. The generated datasets are intentionally ignored by Git via `datasets/`.

## Documentation Assets

The static documentation under `docs/` includes generated site assets and demo audio files. Third-party JavaScript and CSS assets under `docs/site_libs/` retain their upstream license notices.
