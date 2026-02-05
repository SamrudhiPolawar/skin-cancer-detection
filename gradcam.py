def generate_gradcam(img_path, output_path):
    import tensorflow as tf
    import cv2
    import numpy as np

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("conv5_block3_out").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    # FIXED LINE ✔
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, overlay)
