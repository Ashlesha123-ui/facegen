import os
import tensorflow as tf

trained_checkpoint_prefix = ('/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210915-185044/model-20210915-185044.ckpt-100620')
export_dir = os.path.join('export_dir', '0')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph( '/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210915-185044/model-20210915-185044.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()   
