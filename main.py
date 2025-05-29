from fastkaggle import *

comp = 'paddy-disease-classification'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')

from fastai.vision.all import *
set_seed(42)

path.ls()

trn_path = path/'train_images'
files = get_image_files(trn_path)

img = PILImage.create(files[0])
img.to_thumb(128)

from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()

dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)

learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()

learn.lr_find(suggest_funcs=(valley, slide))

learn.fine_tune(3, 0.01)

ss = pd.read_csv(path/'sample_submission.csv')

tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)

probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)

mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
ss['label'] = results
ss.to_csv('subm.csv', index=False)

# if not iskaggle:
#     from kaggle import api
#     api.competition_submit_cli('subm.csv', 'initial rn26d 128px', comp)