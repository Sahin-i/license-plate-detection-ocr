# Source archive note

The following README describes the **incoming archive’s** original split. This project has reconstructed the **trained project split** from `metadata/plates.csv`; use that CSV and `data/` for all training and test commands. Three images moved from each of train and test to restore the project split, with original image pixels and YOLO labels preserved. The provenance and licensing text below is retained from the source archive.

# License plate detection split: 50 train / 10 validation / 25 test

This package combines the previously supplied 70-image Latin-script set with
the **15 UC3M-LP non-Chinese images from Dataset A**. The latter are held out
exclusively for test. All images retain their original pixels and YOLO plate
bounding boxes. Class `0` is `license_plate`.

## Composition

| Primary group | Latin train | Latin val | Latin test | Dataset A test |
| --- | ---: | ---: | ---: | ---: |
| Easy | 18 | 4 | 3 | 0 |
| Medium | 11 | 2 | 2 | 0 |
| Difficult | 7 | 1 | 2 | 0 |
| Multiple vehicles | 8 | 1 | 1 | 10 |
| Partial visibility | 3 | 1 | 1 | 5 |
| Small plate | 3 | 1 | 1 | 0 |
| **Total** | **50** | **10** | **10** | **15** |

The final `data/test/` set has **25 images**: 10 from the Latin set and 15 from
Dataset A. Dataset A images appear nowhere in `train` or `val`. The eight
Mercosur images from the same source parking lot were kept together in train.
Within the other groups, selection was shuffled with seed `20260927`, with
readable OCR examples spread among the splits where available. The splits are
not alphabetical slices.

## Layout

```text
data/
  train/images/   (50)       train/labels/   (50)
  val/images/     (10)       val/labels/     (10)
  test/images/    (25)       test/labels/    (25)
dataset.yaml       YOLO train/val/test paths
annotations.json   per-image pixel boxes, source and OCR metadata
plates.csv         one row for each of the 110 labeled plate boxes
split_manifest.csv one row for each image with split, origin and SHA-256
```

Image names begin with `latin_` or `dataset_a_`, and each image has a label
file of the same stem. The labels contain normalized YOLO detection boxes.
For pixel `[x1, y1, x2, y2]` boxes and original image sizes, use
`annotations.json`. The `origin` and `source_member` fields trace each image
back to its previous package and upstream source.

`dataset.yaml` lists the three image directories relative to the extracted
package root. Run from that root, or set `path` to its absolute location if
your training tool requires it. Keep `test` out of model fitting, model
selection and parameter tuning.

## OCR and evaluation limits

Across all splits there are **110 labeled plate boxes**, of which **44** have
manually verified complete OCR strings: 29 train, 8 val and 7 test. Score exact
match or character error rate only for plate entries where
`ocr_evaluable=true`. An empty `text_gt` is unknown, not an empty true plate.
The **15 Dataset A test images have privacy-masked characters**; they are
appropriate for detection scoring, but none has complete text for OCR exact
match. Their `visible_pattern` values retain the source's `*` placeholders.
The Latin set's partial-visibility group includes cutoffs or deep shadows, not
exclusively physical obstructions. Some scenes may contain additional tiny
plates omitted by the original source labels, so inspect them before treating
this as an exhaustive detection benchmark.

## Checks and provenance

The 85 image files have distinct SHA-256 values. A 64-bit difference-hash
screen found no pair within Hamming distance 8 among these images. Each of the
85 images has one matching label file, and the 110 label rows match the plate
records. No Dataset A image is in train or validation. This checks overlap
inside this package, not overlap with any separate dataset you might train on.

- Latin-set source: **Car License Plate Detection**, distributed by andrewmvd
  on Kaggle (62 images); the distributor lists CC0 1.0. The compilation does
  not provide individual photographer provenance. Source:
  https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- Latin-set source: Silvano et al., **Artificial Mercosur License Plates**,
  Mendeley Data v1 (2020), DOI `10.17632/nx9xbs4rgx.1` (8 images), CC BY 4.0.
  Source: https://data.mendeley.com/datasets/nx9xbs4rgx/1
- Dataset A source: Ramajo-Ballester et al., **UC3M-LP** (15 images from its
  original test split), ODbL 1.0. Source:
  https://github.com/ramajoballester/UC3M-LP ;
  archive: https://zenodo.org/records/17152029 . Check the source's
  attribution and share-alike conditions when redistributing that component.

The Mercosur images retain their source's synthetic plate overlays. No
augmentation was added in this split.
