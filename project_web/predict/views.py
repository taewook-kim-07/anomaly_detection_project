from django.shortcuts import render
from django.http import HttpResponse

import json, os
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit, utils, layers

label_list = ["bottle-broken_large","bottle-broken_small","bottle-contamination","bottle-good","cable-bent_wire","cable-cable_swap","cable-combined","cable-cut_inner_insulation","cable-cut_outer_insulation","cable-good","cable-missing_cable","cable-missing_wire","cable-poke_insulation","capsule-crack","capsule-faulty_imprint","capsule-good","capsule-poke","capsule-scratch","capsule-squeeze","carpet-color","carpet-cut","carpet-good","carpet-hole","carpet-metal_contamination","carpet-thread","grid-bent","grid-broken","grid-glue","grid-good","grid-metal_contamination","grid-thread","hazelnut-crack","hazelnut-cut","hazelnut-good","hazelnut-hole","hazelnut-print","leather-color","leather-cut","leather-fold","leather-glue","leather-good","leather-poke","metal_nut-bent","metal_nut-color","metal_nut-flip","metal_nut-good","metal_nut-scratch","pill-color","pill-combined","pill-contamination","pill-crack","pill-faulty_imprint","pill-good","pill-pill_type","pill-scratch","screw-good","screw-manipulated_front","screw-scratch_head","screw-scratch_neck","screw-thread_side","screw-thread_top","tile-crack","tile-glue_strip","tile-good","tile-gray_stroke","tile-oil","tile-rough","toothbrush-defective","toothbrush-good","transistor-bent_lead","transistor-cut_lead","transistor-damaged_case","transistor-good","transistor-misplaced","wood-color","wood-combined","wood-good","wood-hole","wood-liquid","wood-scratch","zipper-broken_teeth","zipper-combined","zipper-fabric_border","zipper-fabric_interior","zipper-good","zipper-rough","zipper-split_teeth","zipper-squeezed_teeth"];

# load models
model_vgg16 = tf.keras.models.load_model(os.path.dirname(__file__) + '/models/vgg16.h5')
model_vgg16_shape = (1, 224, 224, 3)

model_effnet = tf.keras.models.load_model(os.path.dirname(__file__) + '/models/effnet.h5')
model_effnet_shape = (1, 224, 224, 3)

model_vit = tf.keras.models.load_model(os.path.dirname(__file__) + '/models/vit.h5',
                                            custom_objects={"ClassToken": layers.ClassToken})
model_vit_shape = (1, 224, 224, 3)
# warm-up
with tf.device('/cpu:0'):
    model_vgg16.predict(np.zeros(model_vgg16_shape))
    model_effnet.predict(np.zeros(model_effnet_shape))
    model_vit.predict(np.zeros(model_vit_shape))

def index(request):
    return render(request, "predict/index.html")

def api_predict(request):
    select_model = request.POST.get("select_model")
    file = request.FILES["file"]
    res = preprocessing(select_model, file)
    return HttpResponse(json.dumps({"predict": [res] }))

def binary2cv2(img, image_size):
    img = img.resize((image_size, image_size), Image.HAMMING)
    ret = np.array(img, dtype='float32') / 255.0
    ret = ret.reshape(1, image_size, image_size, 3)
    return ret

def preprocessing(select_model, file):
    file = Image.open(ContentFile(file.read()))

    predict = 0
    if select_model == 'vgg16':
        img = binary2cv2(file, model_vgg16_shape[1])
        predict = model_vgg16.predict(img)
    elif select_model == 'effnet':
        img = binary2cv2(file, model_effnet_shape[1])
        with tf.device('/cpu:0'):
            predict = model_effnet.predict(img)
    elif select_model == 'vit':
        img = binary2cv2(file, model_vit_shape[1])
        with tf.device('/cpu:0'):
            predict = model_vit.predict(img)

    res = label_list[np.argmax(predict)]
    return res