import brambox as bbb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

darknetParser = bbb.io.parser.annotation.DarknetParser(image_dims=[1, 1], class_label_map={0: 'person'})
COCOParser = bbb.io.parser.detection.CocoParser(class_label_map={0: 'person'})

# annotations = darknetParser.parse('inria/Test/pos/yolo-labels/')
annotations = bbb.io.load(darknetParser, 'testing/clean/yolo-labels/')
print(annotations.head())
patch_results = bbb.io.load(COCOParser, 'patch_results.json')
print(patch_results.head())
# patch_up = bbb.io.parse('det_coco', 'patch_up.json', class_label_map={0: 'person'})
clean_results = bbb.io.load(COCOParser, 'clean_results.json')
# noise_results = bbb.io.parse('det_coco', 'noise_results.json', class_label_map={0: 'person'})
# class_results = bbb.io.parse('det_coco', 'class_shift.json', class_label_map={0: 'person'})
# class_only = bbb.parse('det_coco', 'class_only.json', class_label_map={0: 'person'})

plt.figure()
teddy = bbb.stat.pr(patch_results, annotations)
print(teddy)
# up = bbb.stat.pr(patch_up, annotations)['person']
# noise = bbb.stat.pr(noise_results, annotations)['person']
clean = bbb.stat.pr(clean_results, annotations)
# class_shift = bbb.stat.pr(class_results, annotations)['person']
# class_only_pr = bbb.stat.pr(class_only, annotations)['person']



plt.plot(teddy['recall'], teddy['precision'])
#
# plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
#
# ap = bbb.ap(clean[0], clean[1])
# plt.plot(clean[1], clean[0], label=f'CLEAN: AP: {round(ap*100, 2)}%')

# ap = bbb.ap(noise[0], noise[1])
# plt.plot(noise[1], noise[0], label=f'NOISE: AP: {round(ap*100, 2)}%')

# ap = bbb.ap(class_shift[0], class_shift[1])
# plt.plot(class_shift[1], class_shift[0], label=f'OBJ-CLS: AP: {round(ap*100, 2)}%')

# ap = bbb.ap(up[0], up[1])
# plt.plot(up[1], up[0], label=f'OBJ: AP: {round(ap*100, 2)}%')

# ap = bbb.ap(class_only_pr[0], class_only_pr[1])
# plt.plot(class_only_pr[1], class_only_pr[0], label=f'CLS: AP: {round(ap*100, 2)}%')

#plt.gcf().suptitle('PR-curve')
# plt.gca().set_ylabel('Precision')
# plt.gca().set_xlabel('Recall')
# plt.gca().set_xlim([0, 1.05])
# plt.gca().set_ylim([0, 1.05])
# plt.gca().legend(loc=4)
# plt.savefig('pr-curve.eps')
plt.show()
