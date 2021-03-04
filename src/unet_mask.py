# Nuclei Segmentation - UNet using Tensorflow 2
# Intro
# - Dataset used is from NIH ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/NIH-NLM-ThinBloodSmearsPf/
# - The architecture used is [U-Net](https://arxiv.org/abs/1505.04597), which is very common for image segmentation problems such as this.
# - This notebook is inspired from the great kernel [Keras U-net starter - LB 0.277](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277) by Kjetil Åmdal-Sævik.

def unet_predict(model, images):
    ''' returns a unet predicted mask
    model = tensorflow2 model checkpoint .h5
    images = a numpy list np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)'''
    # Predict 
    model = load_model('model_unet_checkpoint.h5')
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions (if prediction )
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test_t)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]), 
                                        (sizes_test[i][0], sizes_test[i][1]), 
                                        mode='constant', preserve_range=True))