from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred):
    """
    Calculer le Dice Coefficient pour une segmentation multi-classe.
    """
    dice_per_class = []
    for c in range(y_true.shape[-1]):  # Itérer sur les classes
        true_c = y_true[..., c]
        pred_c = y_pred[..., c]
        intersection = tf.reduce_sum(true_c * pred_c, axis=[1, 2])
        dice = (2. * intersection) / (
            tf.reduce_sum(true_c, axis=[1, 2]) + tf.reduce_sum(pred_c, axis=[1, 2]) + tf.keras.backend.epsilon()
        )
        dice_per_class.append(dice)

    return tf.reduce_mean(tf.stack(dice_per_class))

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred):
    """
    Calculer le IoU pour une segmentation multi-classe.
    """
    iou_per_class = []
    for c in range(y_true.shape[-1]):  # Itérer sur les classes
        true_c = y_true[..., c]
        pred_c = y_pred[..., c]

        intersection = tf.reduce_sum(true_c * pred_c, axis=[1, 2])
        union = tf.reduce_sum(true_c, axis=[1, 2]) + tf.reduce_sum(pred_c, axis=[1, 2]) - intersection
        iou = intersection / (union + tf.keras.backend.epsilon())
        iou_per_class.append(iou)

    return tf.reduce_mean(tf.stack(iou_per_class))

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    """
    Calculer le Dice Loss pour une segmentation multi-classe.
    """
    smooth = 1e-6
    dice_per_class = []
    for c in range(y_true.shape[-1]):  # Itérer sur les classes
        true_c = y_true[..., c]
        pred_c = y_pred[..., c]
        intersection = tf.reduce_sum(true_c * pred_c)
        union = tf.reduce_sum(true_c) + tf.reduce_sum(pred_c)
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(1 - dice)  # 1 - dice pour obtenir la perte

    return tf.reduce_mean(tf.stack(dice_per_class))

@tf.keras.utils.register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    """
    Combinaison de Binary Crossentropy Loss et Dice Loss pour une segmentation multi-classe.
    """
    bce = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_loss + dice