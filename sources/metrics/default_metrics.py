import tensorflow as tf

DEFAULT_METRICS = ["accuracy",
                   tf.keras.metrics.FalseNegatives(),
                   tf.keras.metrics.FalsePositives(),
                   tf.keras.metrics.TrueNegatives(),
                   tf.keras.metrics.TruePositives(),
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   ]
