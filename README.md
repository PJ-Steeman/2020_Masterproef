# Adversarial YOLO

In dit project hebben we verder gewerkt aan de adversarial patch door S. Thys et. al.

```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

# Instructies
We gebruiken Python 3.6.
Zorg dat u een werkende versie van PyTorch hebt. Meer informatie vindt u hier: https://pytorch.org/

Om het opbouwen van de patch te visualiseren kan u tensorboard installeren:
```
pip install tensorboardX tensorboard
```

Gelieve volgende commando's uit te voeren om de correcte dataset en gewichten te downloaden:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights

```
Indien u gebruik wil maken van de COCO keypoint dataloader en patch applier, dient u de COCO dataset en API te downloaden en installeren, zoals aangegeven op hun website: http://cocodataset.org/#home en GitHub: https://github.com/cocodataset/cocoapi

Nu kan u het volgende commando uitvoeren om het trainen van de patch, die de objectklasse van een persoon in die van een zebra verandert, te beginnen.
Om de gewenste klasse aan te passen moet men in de in train_patch.py op lijn 35 de eerste parameter aanpassen.

```
python3 train_patch.py exp1
```

De code die gebruik maakt van de COCO keypoint dataset staat op dit moment uitgecommentarieerd.
Om de werking hiervan te bekijken moet men het pad in de patch_config.py file aanpassen.
Ook moet men in train_patch.py de dataloader en patch applier aanpassen.
