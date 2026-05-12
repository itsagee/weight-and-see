## Repository structure
 
```
weight-and-see/
├── data/                         # Test images and RGBXY decomposition
├── output/                       # Generated results
├── dithering/                    # Dithering algorithm implementations
│   ├── io.py                     # Loaders for palette, weights, and images (we need a parser for the files generated from RGBXY)
│   ├── floyd_steinberg.py        # Weight-driven Floyd-Steinberg (to be implemented in week 3)
│   └── colour_space.py           # Helper functions to switch between colour spaces
├── metrics/                      # Perceptual and quantitative metrics (to be implemented later)
├── requirements.txt
└── main.py                       # Runs the full pipeline
```
 
## Setup
 
```bash
py -m venv .venv
pip install -r requirements.txt
```