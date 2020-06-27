import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformerKeypointsTensors, PatchApplier
from pycocotools.coco import COCO
from tqdm import tqdm
import json

if __name__ == '__main__':
    print("Setting everything up")
    imgdir = "coco/images/train2017"
    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolo.weights"
    patchfile = "@PJ_PATCHES/final_cat_1s.jpg"

    savedir = "testing_coco"

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformerKeypointsTensors().cuda()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height

    patch_size = 400
    test_size = 100

    clean_results = []
    noise_results = []
    patch_results = []

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    coco = COCO('coco/annotations/person_keypoints_train2017.json')
    cats = coco.loadCats(coco.getCatIds())
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds)
    imgIds = imgIds[:test_size]

    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        I = Image.open('%s/%s'%(imgdir,img['file_name'])).convert("RGB")
        name = os.path.splitext(img['file_name'])[0]
        txtname = name + '.txt'
        txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns_list = coco.loadAnns(annIds)

        keypoint_anns = []
        for x in anns_list:
            # print(x['keypoints'])
            if x['keypoints'][17] > 0 and x['keypoints'][20] > 0:
                keypoint_anns.append(list([0] + x['keypoints']))
        # Hier worden de annotations bijgevuld zodat elke afbeelding er evenveel heeft
        pad_size = max_lab - len(keypoint_anns)
        if(pad_size>0):
            anns = keypoint_anns
            for i in range(pad_size):
                anns.append([1]+[0] * 51)
        else:
            if(pad_size < 0):
                anns = keypoint_anns[:max_lab]
            else:
                anns = keypoint_anns

        w,h = I.size
        if w==h:
            padded_img = I
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(I)
            else:
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(I)

        resize = transforms.Resize((img_size,img_size))
        padded_img = resize(padded_img)

        for i in range (max_lab):
            for j in range (1, 52, 3):
                if w > h:
                    anns[i][j] = anns[i][j]*img_size/w
                    anns[i][j+1] = anns[i][j+1]*img_size/w
                else:
                    anns[i][j] = anns[i][j]*img_size/h
                    anns[i][j+1] = anns[i][j+1]*img_size/h

        cleanname = name + ".png"
        #sla dit beeld op
        padded_img.save(os.path.join(savedir, 'clean/', cleanname))

        boxes = do_detect(darknet_model, padded_img, 0.4, 0.4, True)
        boxes = nms(boxes, 0.4)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id = box[6]
            if(cls_id == 0):   #if person
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                 y_center.item() - height.item() / 2,
                                                                 width.item(),
                                                                 height.item()],
                                      'score': box[4].item(),
                                      'category_id': 1})
        textfile.close()

        anns = torch.Tensor(anns)
        transform = transforms.ToTensor()
        padded_img = transform(padded_img).cuda()
        img_fake_batch = padded_img.unsqueeze(0)
        lab_fake_batch = anns.unsqueeze(0).cuda()

        adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size)
        p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        p_img = p_img_batch.squeeze(0)
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        properpatchedname = name + ".png"
        p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))

        #genereer een label file voor het beeld met sticker
        txtname = properpatchedname.replace('.png', '.txt')
        txtpath = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/', txtname))
        boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
        boxes = nms(boxes, 0.4)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id = box[6]
            if(cls_id == 0):   #if person
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                textfile.write(f'0 {x_center} {y_center} {width} {height}\n')
                patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
        textfile.close()

        #maak een random patch, transformeer hem en voeg hem toe aan beeld
        random_patch = torch.rand(adv_patch_cpu.size()).cuda()
        adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size)
        p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        p_img = p_img_batch.squeeze(0)
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        properpatchedname = name + "_rdp.png"
        p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))

        #genereer een label file voor het beeld met random patch
        txtname = properpatchedname.replace('.png', '.txt')
        txtpath = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/', txtname))
        boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.4, True)
        boxes = nms(boxes, 0.4)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id = box[6]
            if(cls_id == 0):   #if person
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
        textfile.close()

    with open('clean_results.json', 'w') as fp:
        json.dump(clean_results, fp)
    with open('noise_results.json', 'w') as fp:
        json.dump(noise_results, fp)
    with open('patch_results.json', 'w') as fp:
        json.dump(patch_results, fp)
