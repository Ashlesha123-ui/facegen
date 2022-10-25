import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#saved_model_dir="/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210923-110757"
# Convert the model
graph_def_file = "/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210923-110757/saved_model.pb"
import tensorflow as tf
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = "/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210923-110757/saved_model.pb", 
    input_arrays = ['image_batch'],
    output_arrays = ['embeddings'] 
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)

