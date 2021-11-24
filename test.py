from fastbook import *

uploader = widgets.FileUpload()
uploader
img = PILImage.create(uploader.data[0])
_catt,_,probs = learn.predict(img)


uploader = SimpleNamespace(data = ['images/chapter1_cat_example.jpg'])


from fastai.vision.all import *

model = load_learner('plant_disease_model.pkl')

uploader = SimpleNamespace(data = [f'public/disease/{sys.argv[1]}'])
img = PILImage.create(uploader.data[0])
prediction, _, prob = model.predict(img)
prob_value = float(prob.max())*100
