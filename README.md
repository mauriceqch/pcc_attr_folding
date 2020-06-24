# Folding-based compression of point cloud attributes

<p align="center">
  <img src="image.png?raw=true" alt="Folding examples"/>
</p>


* **Authors**:
[Maurice Quach](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Giuseppe Valenzise](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Frederic Dufaux](https://scholar.google.com/citations?user=ziqjbTIAAAAJ)  
* **Affiliation**: Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, 91190 Gif-sur-Yvette, France  
* **Funding**: ANR ReVeRy national fund (REVERY ANR-17-CE23-0020)
* **Links**: [[Paper]](https://arxiv.org/abs/2002.04439)

## Getting started

### Prerequisites

* Python 3.6.9
* TensorFlow 1.15.0 with CUDA 10.0.130, cuDNN 7.4.2 and GCC 7
* [BPG codec](https://bellard.org/bpg/): `bpgenc` and `bpcdec` should be available in your `PATH`
* G-PCC codec [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13): necessary only to compare results with G-PCC,
to obtain more recent versions you may need access to the MPEG Gitlab
* MPEG PCC dataset: refer to Common Test Conditions to download the dataset, you can also get point clouds from [JPEG Pleno](http://plenodb.jpeg.org/).
* packages in `requirements.txt`

*Note*: using a Linux distribution such as Ubuntu is highly recommended

### Configuration

Check the following:
* In `10_pc_to_patch.py`, adjust the patching parameters to your convenience (in particular `N_PATCHES_DEFAULT`).
* In `50_run_mpeg.py`, adjust the configuration parameters to your environment.
* In `91_expdata_full.py`, check that the relative paths are correct for your environment.
The scripts assume there is a single folder containing all the point clouds.
The paths are relative to this folder.

### Compiling Chamfer Distance kernels

We make use of a CUDA implementation for the Chamfer distance.
As such, it is necessary to compile it using the `compile.sh` script and check that the tests run successfully:

    cd ops/nn_distance
    ./compile.sh
    python nn_distance_test.py

You may have to add `-D_GLIBCXX_USE_CXX11_ABI=0` in this script on each `g++` call depending on your compiler and tensorflow version.

## Usage

We provide a set of pipelines to make experimentation easier.
These scripts rely on a YAML that contains information on the experiment: `91_expdata_full.yml` for example.
It is also possible to write your own.

*Important note* : if you wish to reproduce the results of our paper,
we provide the corresponding manual patches in this repository.
To use these patches, copy the folder `manual_patches` and use it as your experimental folder.

For example, you can produce results for folding with the manual patches, produce results for G-PCC
and compare the two with the following commands:

    git clone https://github.com/mauriceqch/pcc_attr_folding.git
    cd pcc_attr_folding/src
    cp -r ../manual_patches ~/data/experiments/pcc_attr_folding_manual
    python 61_run_folding.py 91_expdata_full.yml ~/data/experiments/pcc_attr_folding_manual
    python 50_run_mpeg.py 91_expdata_full.yml ¬/data/experiments/gpcc
    python 70_run_eval_compare.py 91_expdata_full.yml ~/data/experiments/gpcc/ ~/data/experiments/pcc_attr_folding_manual
    
To produce results without division into patches:

    python 61_run_folding.py 91_expdata_full.yml ~/data/experiments/pcc_attr_folding_single --k 1
    python 70_run_eval_compare.py 91_expdata_full.yml ~/data/experiments/gpcc/ ~/data/experiments/pcc_attr_folding_single

### Batch folding pipeline

Runs the folding pipeline for each point cloud.

    python 61_run_folding.py 91_expdata_full.yml ~/data/experiments/pcc_attr_folding
    
### Folding pipeline (single point cloud)

Runs the complete folding pipeline for a point cloud:
* Divides point cloud into k patches
* Fits a grid for each patch
* Evaluates compression for each patch
* Merge the patches and compression results.

For example, to run the pipeline while divinding the point cloud into 4 patches:

    python 60_folding_pipeline.py loot.ply loot/ --k 4

### MPEG G-PCC

We provide a script to run MPEG G-PCC experiments.

    python 50_run_mpeg.py 91_expdata_full.yml gpcc

### Evaluate and compare

To compare evaluation results between GPCC and our method.

    python 70_run_eval_compare.py 91_expdata_full.yml ~/data/experiments/gpcc/ ~/data/experiments/pcc_attr_folding/

Also, to compare with two versions of GPCC as in the paper.

    python 72_run_eval_compare_two.py 91_expdata_full.yml ~/data/experiments/gpcc/ ~/data/experiments/gpcc-v3/ ~/data/experiments/pcc_attr_folding
    
## Overview

    ├── manual_patches                      [Data] Manual patches used in the paper
    ├── requirements.txt                    Package requirements
    └── src
        ├── 10_pc_to_patch.py               [Preprocess] Divide a point cloud into patches
        ├── 11_train.py                     [Train] Fold a grid onto a point cloud and save the obtained network
        ├── 12_merge_ply.py                 [Preprocess] Merge point cloud patches into a single point cloud
        ├── 20_gen_folding.py               [Inference] Generate a folded grid with a trained network
        ├── 21_eval_folding.py              [Eval] Refine, optimize, compress and evaluate a folded grid
        ├── 22_merge_eval.py                [Eval] Merge evaluation results at different QPs
        ├── 23_eval_merged.py               [Eval] Evaluates a merged point cloud compared to the original
        ├── 50_run_mpeg.py                  [MPEG] Runs G-PCC on all point clouds
        ├── 51_gen_report.py                [MPEG] Parse files in a G-PCC result folder and generate a JSON report
        ├── 60_folding_pipeline.py          [Folding] Runs the full folding pipeline (patches, folding, compress, eval)
        ├── 61_run_folding.py               [Folding] Runs the full folding pipeline for all point clouds
        ├── 70_run_eval_compare.py          [Eval] Compare results for all point clouds with G-PCC
        ├── 71_eval_compare.py              [Eval] Compare results with G-PCC
        ├── 72_run_eval_compare_two.py      [Eval] Compare results for all point clouds with two G-PCC result folders for all point clouds
        ├── 73_eval_compare_two.py          [Eval] Compare results with two G-PCC result folders
        ├── 80_input.py                     [Model] Input pipeline for the network
        ├── 80_model.py                     [Model] Model for the network
        ├── 90_run_tests.py                 Run tests
        ├── 91_ds_expdata.py                [Utils] Downsample all point clouds
        ├── 91_expdata_full.yml             [Config] Experimental data, list of point clouds considered
        ├── 98_highlight_borders.py         [Utils] Highlight borders on a folded grid (debugging)
        ├── 994_pc_curvature.py             [Utils] Compute point cloud curvature (debugging)
        ├── 99_pc_to_vg.py                  [Utils] Voxelize a point cloud
        ├── 99_pc_to_vg_batch.py            [Utils] Voxelize multiple point clouds
        ├── ops                             Chamfer distance files
        └── utils
            ├── adj.py                      Folded grid refinement
            ├── bd.py                       BD-RATE/BD-PSNR
            ├── bpg.py                      BPG compression (BPG is a image compression codec based on HEVC intra)
            ├── color_mapping.py            Color mapping, used for transferring colors from the grid to the point cloud and vice-versa
            ├── color_space.py              Color space conversion
            ├── curvature.py                Curvature computation
            ├── generators.py               Generators for data pipelines
            ├── grid.py                     Grid manipulation
            ├── mpeg_parsing.py             MPEG log files parsing
            ├── parallel_process.py         Parallel processing
            ├── pc_io.py                    Point Cloud Input/Output
            └── quality_eval.py             Point Cloud color distortion metrics

## Invididual Usage Examples

It is possible to use the scripts individually instead of using the pipelines.
We provide some usage examples below.

### Training

#### Point cloud to patches

    python 10_pc_to_patch.py loot.ply loot_patches/ --k 9
    
#### Fitting a grid on a point cloud

    python 11_train.py loot.ply loot_model/ --max_steps 2000 --model 80_model --input_pipeline 80_input --grid_steps 64,128,1
   
#### Merging patches into a point cloud

    python 12_merge_ply.py loot_patches/*.ply loot_merged.ply
    
### Evaluation

#### Evaluate color compression

    python 20_gen_folding.py loot.ply loot_results/ loot_model/ --model 80_model --input_pipeline 80_input --grid_steps auto
    python 21_eval_folding.py loot_results/
    python 23_eval_merged.py loot.ply loot_results/refined_opt_qp_20/loot_remap.ply

#### Compare evaluation results

    python 29_eval_compare.py gpcc/octree-predlift/lossless-geom-lossy-attrs/Egyptian_mask_vox12 ./test_egypt/merged/

### Utilities
    
#### Downsample experimental data

Create a downsampled version of experimental data for a given setup.
This only works for point cloud with `voxXX` in their name such as `longdress_vox10_1200`.

    python 91_ds_expdata.py 91_expdata_full.yml 8

#### Highlight borders

Given a folded point cloud, highlights the borders of the grid on the point cloud.

    python 98_highlight_borders.py carpet_folded.ply carpet_folded_with_borders.ply

#### Batch downsample point clouds

Given a pattern, downsample all matching point clouds.

    python 99_pc_to_vg_batch.py "pcs_vox10/**/*.ply" pcs_vox_08 --vg_size 256

## Citation

    @article{DBLP:journals/corr/abs-2002-04439,
      author    = {Maurice Quach and
                   Giuseppe Valenzise and
                   Fr{\'{e}}d{\'{e}}ric Dufaux},
      title     = {Folding-based compression of point cloud attributes},
      journal   = {CoRR},
      volume    = {abs/2002.04439},
      year      = {2020},
      url       = {https://arxiv.org/abs/2002.04439},
      archivePrefix = {arXiv},
      eprint    = {2002.04439},
      timestamp = {Wed, 12 Feb 2020 16:38:55 +0100},
      biburl    = {https://dblp.org/rec/journals/corr/abs-2002-04439.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
