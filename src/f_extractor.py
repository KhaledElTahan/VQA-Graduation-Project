import tensorflow as tf

_IS_CREATED = False
_RES_NET_SESS = None
_RES_NET_GRAPH = None

def _create_graph():
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph("./../models/resnet152/ResNet-L152.meta")

    # requires a session in which the graph was launched.
    new_saver.restore(sess, "./../models/resnet152/ResNet-L152.ckpt")

    graph = tf.get_default_graph()

    global _IS_CREATED, _RES_NET_SESS, _RES_NET_GRAPH
    _IS_CREATED = True
    _RES_NET_SESS = sess
    _RES_NET_GRAPH = graph

def _get_features_tensor():
    return _RES_NET_GRAPH.get_tensor_by_name("avg_pool:0")

def _get_images_tensor():
    return _RES_NET_GRAPH.get_tensor_by_name("images:0")

def _close_session():
    global _IS_CREATED, _RES_NET_SESS, _RES_NET_GRAPH

    if _IS_CREATED and _RES_NET_SESS:
        _RES_NET_SESS.close()
        _IS_CREATED = False
        _RES_NET_GRAPH = None
        _RES_NET_SESS = None

# accepts list of images, each image of 224x224x3
def get_features(batch):
    if not _IS_CREATED:
        _create_graph()

    feed_dict = {_get_images_tensor(): batch}
    return _RES_NET_SESS.run(_get_features_tensor(), feed_dict=feed_dict)


