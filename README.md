# Weight & See
This repository contains the code developed as part of a research internship at the University of Groningen, conducted in collaboration with Adobe Research. The project focuses on reconstructing images using dithering and a palette of limited colours per image, as well as evaluating the resulting images, in terms of similarity with the original.

The instructions below describe how to set up the environment and run the code.
 
## Setup
 
In order to run the code, we need to set up a few things first. You'll need to set up and activate a venv. To do that and install the needed dependencies, from the root folder run the following commands:

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### LPIPS and DreamSim
For this project, we use [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [DreamSim](https://github.com/ssundaram21/dreamsim) as  part of the image similarity metrics. Both packages, together with their required dependencies are included in the `requirements.txt`, so no separate installation is required.

LPIPS and DreamSim will automatically download their pretrained model weights on the first time the code is run. An internet connection is therefore required on the first run that uses these metrics. Later runs should use the locally cached model weights.

## App
To run the app GUI, run the following command from the root directory for a single run:
```
python app.py
```
or the following that auto-reloads with every file save, in case you are making changes to the code:
```
gradio app.py
```

When running for a new image, make sure you have the palette.js and the weights.js files for the image, as generated from the [fastlayers implementation](https://cragl.cs.gmu.edu/fastlayers/). In this repository, other than for the `blue-gradient` image, there are no `weight.js` files, due to their large size.

## Gallery
Every time you have new resutls and want to add them to the gallery view, and for the first time running this code, run the following code from the root directory to generate the `gallery.thml`:

```
python build_gallery.py--output ./output
```
or just
```
python build_gallery.py
```

With this, a `gallery.html` file will be generate including all the results from the `output` directory will. 

## All Metrics
To concatenate all metric results for all images in the `output` directory, run one of the following commands from the root directory:
```
python concat_metrics.py --output ./output --dest ./all_metrics.csv
```
or the following, to see which image resutls have been discovered and included:
```
python concat_metrics.py --output ./output --dest ./all_metrics.csv --verbose
```
or alternatively, to get a summary of the data included in the folder:
```
    python concat_metrics.py --output ./output --dest ./all_metrics.csv --verbose --summary
```

## Repository structure
 
```
weight-and-see/
├── data/                           # Test images and RGBXY decomposition
│   ├── [image-name].jpg/.png       # Original test images
│   └── [image-name]/
│       ├── [image-name]-palette.js # Palette colour file
│       └── [image-name]-weights.js # This is where per-pixel mixing weights should go, not included here because of size
│
├── dithering/                   # Dithering algorithm implementations
│   ├── floyd_steinberg.py       # Floyd-Steinberg implementation
│   ├── jarvis_judice_ninke.py   # Jarvis-Judice-Ninke implementation
│   ├── stucki.py                # Stucki implementation
│   ├── io.py                    # Loaders for palette, weights, and images (we need a parser for the files generated from RGBXY)
│   ├── dithering_utils.py           # Helper functions for dithering
│   └── colour_space_conversions.py  # Helper functions to switch between colour spaces
│
├── metrics/             # Perceptual and quantitative metric implementations
│   ├── psnr.py          # PSNR implementation
│   ├── ssim.py          # SSIM implementation
│   ├── ciede2000.py     # CIEDE2000 implementation
│   ├── s_cielab.py      # S-CIELAB implementation
│   ├── lpips.py         # LPIPS implementation
│   └── dreamsim.py      # DreamSim implementation
│
├── models/       # Models for the metric models should go here
│
├── runners/
│   ├── dithering_runner.py    # used to identify and call the right dithering function as well as save the results
│   ├── metrics_runner.py      # used to identify and call the right metric as well as to save the results
│   └── dithering_viz.py       # includes all functionality for the visualization plots and figures
│
├── output/                    # Generated results
│   └── [image-name]
│       ├── fs/         # directory including all the FS results for this image
│       ├── jjn/        # directory including all the JJN results for this image
│       ├── st/         # directory including all the ST results for this image
│       ├── [image-name]_fs_results.npz    # npz file for all the FS results
│       ├── [image-name]_jjn_results.npz   # npz file for all the JJN results
│       ├── [image-name]_st_results.npz    # npz file for all the ST results
│       └── metrics/    # directory including all the metric results (plots and csv)
│           ├── *.png                           # metric resulting plots & figures
│           └── [image-name]-metrics_all.csv    # csv files for all metric results
│
├── requirements.txt          # used to set up once before running the code
├── refactored-main.ipynb     # The most up-to-date code that can run the full pipeline or individual sections, clean and tidy.
├── app.py                    # The **main** code and its GUI, user-friendly.
├── build_gallery.py          # Compiles the gallery and creates the html file
├── gallery.html              # this is where the gallery.html will be when created
├── concat_metrics.py         # used to find and concat all the individual images results into one single file
├── all_metrics.csv           # one single csv file including all the results for all the images
└── main.ipynb                # Old code used to develop and test individual images
```