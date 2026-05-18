# Demo page specification

This document describes the specification of demo page.

## Specification

- The demo page is defined with /home/yfujita/research/marse/docs/index.qmd and rendered with quorto
- The demo page has "Abstract" section, "Method" section, and "Audio example" section 
- The "Audio example" present noisy, clean, and enhanced audio samples with multiple experiments for each sample
- Users can select audio sample id with pull down UI 
- When an audio sample id is selected, audio players for the selected audio sample id are displayed in the arrangement below:
    1. Noisy speech, ground-truth clean speech, enhanced speech of cnar_N1, enhanced speech of c_ar_N50, c_mar_discriminative_N1
    2. c_mar_autoregressive_N5, c_mar_autoregressive_N10, c_mar_autoregressive_N20, c_mar_autoregressive_N30, c_mar_autoregressive_N40, c_mar_autoregressive_N50
    3. c_mar_random_N5, c_mar_random_N10, c_mar_random_N20, c_mar_random_N30, c_mar_random_N40, c_mar_random_N50
    4. c_mar_oracle_N5, c_mar_oracle_N10, c_mar_oracle_N20, c_mar_oracle_N30, c_mar_oracle_N40, c_mar_oracle_N50

## Structure of `audiosamples` file in each experimental directories

```
/<experimental directory name>
    /audiosamples
        /<sample name>
            /clean.wav : ground-truth clean speech
            /noisy.wav : noisy speech
            /reconstructed.wav : enhanced speech with the experiment
```

## Paths for experiments on samples from Libri1Mix dataset

### Experimental directory paths
outputs/archives/marse/labeled_libri_mix_ruche/cnar_exp-001/ruche-v2/iwaenc2026_cnar_N1
outputs/archives/marse/labeled_libri_mix_ruche/c_ar/2026-03-10_22-26-39/iwaenc2026_c_ar_N50
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_discriminative_N1
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N5
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N10
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N20
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N30
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N40
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/mgse_c_mar_autoregressive_N50
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N5
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N10
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N20
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N30
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N40
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_random_N50
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N5
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N10
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N20
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N30
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N40
outputs/archives/marse/labeled_libri_mix_ruche/c_mar/2026-03-10_17-16-49/iwaenc2026_c_mar_oracle_N50

### Audio sample ids
Libri2Mix_test_1089-134686-0007_7021-85628-0018
Libri2Mix_test_1089-134691-0019_672-122797-0023
Libri2Mix_test_1089-134686-0024_1995-1836-0013
Libri2Mix_test_1089-134691-0014_7127-75947-0023
Libri2Mix_test_1284-1180-0020_8555-284447-0005
Libri2Mix_test_1221-135766-0003_1320-122617-0009
Libri2Mix_test_1284-1180-0032_2830-3979-0003
Libri2Mix_test_1284-1180-0005_6829-68771-0000
Libri2Mix_test_1284-1180-0014_6829-68769-0006
Libri2Mix_test_121-127105-0019_4446-2275-0029

## Paths for experiments on samples from LibriDEMAND dataset

### Experimental directory paths
outputs/cnar_exp-001/ruche-v2/mgse_demand_cnar_N1
outputs/c_ar/2026-03-10_22-26-39/mgse_demand_c_ar_N50
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_discriminative_N1
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N5
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N10
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N20
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N30
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N40
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_autoregressive_N50
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N5
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N10
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N20
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N30
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N40
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_random_N50
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N5
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N10
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N20
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N30
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N40
outputs/c_mar/2026-03-10_17-16-49/mgse_demand_c_mar_oracle_N50

### Audio sample ids
Libri1Mix_test_1363-135842-0000
Libri1Mix_test_1898-145720-0018
Libri1Mix_test_441-130108-0041
Libri1Mix_test_8238-283452-0035
Libri1Mix_test_3699-175950-0005
Libri1Mix_test_27-124992-0056
Libri1Mix_test_229-130880-0014
Libri1Mix_test_2952-410-0016
Libri1Mix_test_6078-54013-0044
Libri1Mix_test_201-122255-0029