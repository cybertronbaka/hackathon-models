from fastai.vision.all import *
path = Path('./')
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224))
#path.ls()[0].ls()
learner = cnn_learner(dls, resnet34, metrics=error_rate)
dls.valid_ds.items[:3]
learner.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learner.export()
path = Path()
path.ls(file_exts='.pkl')

