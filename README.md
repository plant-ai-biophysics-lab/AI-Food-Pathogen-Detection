# AI-Enabled Biosensing for Rapid Identification of Pathogens in Food and Agricultural Water

We provide the Faster R-CNN implementation to quantify bacterial pathogens in food and agricultural water using images acquired by phage-based biosensing.

**Paper Link (to be updated)**

## Install Requirements

```
conda env create -f environment.yml -n ai-food-pathogen
```

## Prepare Data


### **Datasets[^1]**

- **mono**: Images for *Escherichia coli* monoculture
- **non-ecoli**: Images for selected non-*E. coli* bacteria[^2]
- **labmix**: Images for microbial mixture of *E. coli* and non-*E. coli* bacteria
- **ccw**: Images for coconut water sample inoculated with *E. coli*
- **spw**: Images for spinach wash water sample inoculated with *E. coli*
- **irw**: Images for irrigation water sample inoculated with *E. coli*/ enriched using generic media
- **irwEC**: Images for irrigation water sample inoculated with *E. coli*/ enriched using selective media (EC broth)

[^1]: Initial loads used for *E. coli* inoculation: 10 CFU/mL (low), 10<sup>2</sup> CFU/mL (high2), 10<sup>3</sup> CFU/mL (high3).
[^2]: *Listeria innocua*, *Bacillus subtilis*, and *Pseudomonas fluorescens*

### **Data organization**

- File names should be in the format of 'Acquired-(int).jpg'

Data used to train Faster R-CNN model
```bash
ai-food-pathogen-data
├── train
│   ├── mono-sp
│   │   ├── mono_1
│   │   ├── mono_2
│   │   └── mono_3
│   ├── mono_1
│   ├── mono_2-3
│   ├── non-ecoli
│   ├── labmix_1
│   └── labmix_2-3
├── val
│   ├── mono-sp
│   │   ├── mono_1
│   │   ├── mono_2
│   │   └── mono_3
│   ├── mono_1
│   ├── mono_2-3
│   ├── non-ecoli
│   ├── labmix_1
│   └── labmix_2-3
├── mono_1_annotations.xml
├── mono_2-3_annotations.xml
├── labmix_0_annotations.xml
├── labmix_1_annotations.xml
└── labmix_2-3_annotations.xml

```

Data used to test and evalute the model
```bash
ai-food-pathogen-data
├── test
│   ├── ccw
│   ├── spw
│   ├── irw
│   └── irwEC
├── ccw_annotations.xml
├── spw_annotations.xml
├── irw_annotations.xml
└── irwEC_annotations.xml

```
`dataset.py`

## Training

`train.py`

## Evaluation

`eval_mono.py`
`eval.py`

## Acknowledgements

The Faster R-CNN implementation is from *Shaoqing Ren*

https://github.com/ShaoqingRen/faster_rcnn

## Acknowledgements

This project was supported by the USDA-NIFA (Grant 2021-67021-34256) and the [National AI Institute for Food Systems (AIFS)](https://aifs.ucdavis.edu).
