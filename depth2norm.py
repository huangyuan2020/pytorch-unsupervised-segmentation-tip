import cv2
import numpy as np
import mxnet as mx
def depth2normal(depth):
    h, w=depth.shape
    dx = -(depth[1:h-1, 2:w] - depth[1:h-1, 0:w-2])*0.5
    dy = -(depth[2:h, 1:w-1] - depth[0:h-2, 1:w-1])*0.5
    dz = mx.nd.ones((h-2,w-2))
    dl = mx.nd.sqrt(mx.nd.elemwise_mul(dx, dx) + mx.nd.elemwise_mul(dy, dy) + mx.nd.elemwise_mul(dz, dz))
    dx = mx.nd.elemwise_div(dx, dl) * 0.5 + 0.5
    dy = mx.nd.elemwise_div(dy, dl) * 0.5 + 0.5
    dz = mx.nd.elemwise_div(dz, dl) * 0.5 + 0.5
    return np.concatenate([dy.asnumpy()[np.newaxis,:,:],dx.asnumpy()[np.newaxis,:,:],dz.asnumpy()[np.newaxis,:,:]],axis=0)


if __name__ == '__main__':
    depth=cv2.imread("./interior/depth/0.png", -1)
    normal=np.array(depth2normal(mx.nd.array(depth))*255)
    normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)         
    cv2.imwrite("normal.png",normal.astype(np.uint8))