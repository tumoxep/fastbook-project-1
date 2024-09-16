from fastai import learner
from fastai.vision.core import PILImage
import gradio as gr

learn_inf = learner.load_learner('export.pkl')

def image_classifier(inp):
    pred, pred_idx, probs = learn_inf.predict(inp)
    return pred, float(probs[pred_idx])

demo = gr.Interface(fn=image_classifier, allow_flagging="never", inputs="image", outputs=[gr.Label(label="Prediction"), gr.Label(label="Probability")])
demo.launch()
