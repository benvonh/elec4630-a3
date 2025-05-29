# =========================================================
# 0. Setup
# =========================================================
from fastkaggle import *
comp = 'paddy-disease-classification'
path = setup_comp(comp, install='fastai "timm>=0.9.12.dev0"')

from fastai.vision.all import *
from sklearn.model_selection import StratifiedGroupKFold
import warnings, random, numpy as np, torch
set_seed(42, reproducible=True)
warnings.filterwarnings('ignore')

# =========================================================
# 1. Data – group-aware split to avoid leakage
# =========================================================
df = pd.read_csv(path/'train.csv')           # meta csv supplied by Kaggle
df['fname'] = df['image_id'].apply(lambda o: f'train_images/{o}')
splits = StratifiedGroupKFold(
            n_splits = 5, shuffle=True, random_state=42
         ).split(df, df['label'], groups=df['variety'])   # ⬅ group key

train_idx, valid_idx = next(splits)         # first fold is fine

item_tfms = Resize(256, method='squish')    # starter size
batch_tfms = aug_transforms(size=256,min_scale=.6,max_rotate=20,
                            max_zoom=1.2,max_lighting=.2,max_warp=.2,
                            p_lighting=.9, p_affine=.9) + [
                RandomErasing(p=0.4,max_count=2),
                Normalize.from_stats(*imagenet_stats)
             ]

dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x = ColReader('fname', pref=path),
        get_y = ColReader('label'),
        splitter = IndexSplitter(valid_idx),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
      ).dataloaders(df, bs=64)

# =========================================================
# 2. Model – ConvNeXt Small pretrained on ImageNet-22k
# =========================================================
arch = 'convnext_small_in22k'   # needs timm>=0.9.2
learn = vision_learner(dls, arch,
                       metrics=accuracy,
                       loss_func=LabelSmoothingCrossEntropy(eps=0.1),
                       wd=0.01).to_fp16()

# =========================================================
# 3. Training – progressive resizing & discriminative LR
# =========================================================
def fine_tune_size(sz, epochs, lr, freeze_epochs=1):
    dls.item_tfms = Resize(sz, method='squish')
    dls.batch_tfms[0].size = sz            # RandomResizedCrop inside aug list
    learn.freeze()
    learn.fit_one_cycle(freeze_epochs, lr/10)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr)

sched = {256:(1,3,2e-3), 384:(1,3,1e-3), 480:(0,3,5e-4)}
for sz,(fz,ep,lr) in sched.items():
    fine_tune_size(sz, ep, lr, freeze_epochs=fz)

# =========================================================
# 4. Inference – TTA + CutMix disabled
# =========================================================
# Switch off heavy aug for test
dls.item_tfms = Resize(480, method='squish')
dls.batch_tfms = [Normalize.from_stats(*imagenet_stats)]

tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)
probs,_ = learn.tta(dl=tst_dl, n=4, beta=0.12)  # 4-crop TTA

ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = probs.argmax(dim=1).map(dict(enumerate(dls.vocab)))
ss.to_csv('subm.csv', index=False)

# Kaggle submission
# api.competition_submit_cli('subm.csv', 'convnext-s prog-resize', comp)
