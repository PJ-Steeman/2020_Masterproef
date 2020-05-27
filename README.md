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
Nu kan u het volgdene commando uitvoeren om het trainen van de patch te beginnen

```
python3 train_patch.py paper_obj
```
