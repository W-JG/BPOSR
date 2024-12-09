from os import path as osp
import os
import shutil
import glob as gb
from PIL import Image

# refernce: https://github.com/Fanghua-Yu/OSRT/blob/master/odisr/utils/make_clean_lau_dataset.py

src = './Dataset/lau_dataset'
dst = './Dataset/lau_dataset_clean'
os.makedirs(dst, exist_ok=True)
train_dict = {'016': 'part of ERP', '026': 'virtual scenario', '027': 'virtual scenario', '028': 'virtual scenario',
              '029': 'virtual scenario', '030': 'virtual scenario', '031': 'virtual scenario',
              '032': 'virtual scenario',
              '033': 'virtual scenario', '034': 'virtual scenario', '035': 'virtual scenario',
              '040': 'virtual scenario',
              '051': 'virtual scenario', '053': 'virtual scenario', '054': 'virtual scenario',
              '055': 'virtual scenario',
              '057': 'virtual scenario', '060': 'virtual scenario', '124': 'mistakes in transform',
              '131': 'virtual scenario',
              '169': 'virtual scenario', '320': 'mistakes in transform', '321': 'mistakes in transform',
              '322': 'mistakes in transform', '329': 'mistakes in transform', '354': 'mistakes in transform',
              '360': 'mistakes in transform', '361': 'mistakes in transform', '362': 'mistakes in transform',
              '391': 'mistakes in transform', '409': 'mistakes in transform', '453': 'mistakes in transform',
              '458': 'mistakes in transform', '506': 'virtual scenario', '522': 'mistakes in transform', '541':
                  'part of ERP', '554': 'extremely poor quality', '559': 'extremely poor quality',
              '634': 'extremely poor quality',
              '644': 'extremely poor quality', '676': 'mistakes in transform', '813': 'mistakes in transform',
              '853': 'virtual scenario', '915': 'virtual scenario', '1049': 'mistakes in transform',
              '1055': 'mistakes in transform', '1072': 'virtual scenario', '1177': 'virtual scenario',
              '1184': 'mistakes in transform', '1196': 'extremely poor quality',
                '642': 'size error', '024': 'size error'
              }

test_dict = {}
validation_dict = {}
suntest_dict = {}

         
_dict = {'odisr/training/HR': train_dict, 'odisr/training/LR/X4': train_dict, 'odisr/training/LR/X8': train_dict,'odisr/training/LR/X16': train_dict,
         'odisr/testing/HR': test_dict,'odisr/testing/LR/X4': test_dict, 'odisr/testing/LR/X8': test_dict,'odisr/testing/LR/X16': test_dict,
         'odisr/validation/HR': validation_dict, 'odisr/validation/LR/X4': validation_dict, 'odisr/validation/LR/X8': validation_dict,'odisr/validation/LR/X16': validation_dict,
         'sun_test/HR': suntest_dict,'sun_test/LR/X4': suntest_dict, 'sun_test/LR/X8': suntest_dict,'sun_test/LR/X16': suntest_dict,
         
         }

for split_type, rm_dict in _dict.items():
    img_paths = gb.glob(src + '/%s/*' % split_type)
    for i, img_path in enumerate(img_paths):
        img_idx = osp.splitext(osp.split(img_path)[1])[0]
        relative_path = img_path.split('lau_dataset/')[-1]
        if img_idx in rm_dict.keys():
            print('rm %s: %s' % (img_idx, rm_dict[img_idx]))
            continue
        try:
            shutil.copy(img_path, osp.join(dst, relative_path))
        except FileNotFoundError:
            os.makedirs(osp.split(osp.join(dst, relative_path))[0], exist_ok=True)
            shutil.copy(img_path, osp.join(dst, relative_path))
        if 'jpg' in relative_path:
            os.rename(osp.join(dst, relative_path), osp.join(dst, relative_path[:-3] + 'png'))
        print('[%s][%s/%s]' % (split_type, i, len(img_paths)))
