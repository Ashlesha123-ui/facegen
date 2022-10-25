import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#gf = tf.GraphDef()   
#m_file = open("/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210923-110757/saved_model.pb",'rb')
#import tensorflow as tf
Graph = tf.GraphDef()   
File = open("/home/prixgen-gpu/Desktop/FaceGen/facenet/src/models/20210923-110757/saved_model.pb","rb")
Graph.ParseFromString(File.read())

for Layer in Graph.node:
    print(Layer.name)