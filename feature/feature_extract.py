from utils.extractor import Extractor
from models.vgg import vgg16
from models.sketch_resnet import resnet50
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_set_root = '/Users/minglirui/gpt/SketchyDatabase/dataset/sketch-triplet-train'
test_set_root = '/Users/minglirui/gpt/SketchyDatabase/dataset/sketch-triplet-test'

train_photo_root = '/Users/minglirui/train_data/rendered_256x256/256x256/photo/tx_000000000000/airplane'
test_photo_root = '/Users/minglirui/train_data/rendered_256x256/256x256/photo/tx_000000000000/airplane'

# The trained model root for resnet
SKETCH_RESNET = '/Users/minglirui/gpt/SketchyDatabase/mixed_triplet_loss/sketch/sketch_resnet_85.pth'
PHOTO_RESNET = '/Users/minglirui/gpt/SketchyDatabase/mixed_triplet_loss/photo/photo_resnet_85.pth'

# The trained model root for vgg
SKETCH_VGG = '/Users/minglirui/gpt/SketchyDatabase/vgg_triplet_loss/sketch/sketch_vgg_190.pth'
PHOTO_VGG = '/Users/minglirui/gpt/SketchyDatabase/vgg_triplet_loss/photo/photo_vgg_190.pth'

FINE_TUNE_RESNET = '/Users/minglirui/gpt/SketchyDatabase/fine_tune/model_270.pth'

device = 'cuda:1'

'''vgg'''
vgg = vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')))
vgg.cuda()

ext = Extractor(pretrained=False)
ext.reload_model(vgg)

photo_feature = ext.extract_with_dataloader(test_photo_root, 'photo-vgg-190epoch.pkl')

vgg.load_state_dict(t.load(SKETCH_VGG, map_location=t.device('cpu')))
ext.reload_model(vgg)

sketch_feature = ext.extract_with_dataloader(test_set_root, 'sketch-vgg-190epoch.pkl')


'''resnet'''
resnet = resnet50()
resnet.fc = nn.Linear(in_features=2048, out_features=125)
resnet.load_state_dict(t.load(PHOTO_RESNET, map_location=t.device('cpu')))
resnet.cuda()

ext = Extractor(pretrained=False)
ext.reload_model(resnet)

photo_feature = ext.extract_with_dataloader(test_photo_root, 'photo-resnet-epoch.pkl')

resnet.load_state_dict(t.load(SKETCH_RESNET, map_location=t.device('cpu')))
ext.reload_model(resnet)

sketch_feature = ext.extract_with_dataloader(test_set_root, 'sketch-resnet-epoch.pkl')

