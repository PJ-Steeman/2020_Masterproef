import brambox as bbb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

darknetParser = bbb.io.parser.annotation.DarknetParser(image_dims=[1, 1], class_label_map={0: 'person'})
COCOParser_clean = bbb.io.parser.detection.CocoParser(class_label_map={0: 'person'})
COCOParser_noise = bbb.io.parser.detection.CocoParser(class_label_map={0: 'person'})
COCOParser_patch = bbb.io.parser.detection.CocoParser(class_label_map={0: 'person'})

annotations = bbb.io.load(darknetParser, 'testing_coco/clean/yolo-labels/')

clean_results = bbb.io.load(COCOParser_clean, 'clean_results.json')
noise_results = bbb.io.load(COCOParser_noise, 'noise_results.json')
patch_results = bbb.io.load(COCOParser_patch, 'patch_results.json')

plt.figure()
clean = bbb.stat.pr(clean_results, annotations)
noise = bbb.stat.pr(noise_results, annotations)
patch = bbb.stat.pr(patch_results, annotations)

clean_ap = bbb.stat.ap(clean)
noise_ap = bbb.stat.ap(noise)
patch_ap = bbb.stat.ap(patch)

plt.plot(clean['recall'], clean['precision'], label='CLEAN')
plt.plot(noise['recall'], noise['precision'], label=f'NOISE: AP: {round(noise_ap*100, 2)}%')
plt.plot(patch['recall'], patch['precision'], label=f'PATCH: AP: {round(patch_ap*100, 2)}%')

plt.gcf().suptitle('PR-curve')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)

plt.show()
