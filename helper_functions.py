'''

'''


import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
K.clear_session()
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from sklearn.utils.extmath import cartesian
import math
import numpy as np

smooth = 1.
dropout_rate = 0.5
act = "relu"

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def bce_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)

# linear transformation for the Fourier descriptors
def linear_transform(X):
  #b, m, n = X.shape.as_list()
  b = 8
  m = 1
  n = 256
  w_arr = np.linspace(1, n, num=n)
  w_arr = np.reshape(w_arr, (m,n))
  w_arr = np.tile(w_arr, (b, m, 1))
  #w_arr = sigmoid(w_arr)
  w = tf.constant(w_arr, shape = (b,m,n))
  w = tf.cast(w, tf.complex64)
  return X * w
  
def sigmoid(X):
   return 1/(1+np.exp(-X))
   
def high_on_low_f(X):
  b = 8
  m = 1
  n = 256
  x = np.linspace(5, -3, num=128)
  # sigmoid weight between 1 and 0, high on low frequency
  # input X 0-64 and 65-128
  weight = sigmoid(x)
  weight = np.append(weight,weight[::-1])
  w_arr = np.reshape(weight, (m,n))
  w_arr = np.tile(w_arr, (b, m, 1))
  w = tf.constant(w_arr, shape = (b,m,n))
  w = tf.cast(w, tf.complex64)
  return X * w

def high_on_low_f2(X):
  b = 8
  m = 1
  n = 256
  x = np.linspace(4, -4, num=128)
  # sigmoid weight between 1 and 0, high on low frequency
  # input X 0-64 and 65-128
  weight = sigmoid(x)
  weight = np.append(weight,weight[::-1])
  w_arr = np.reshape(weight, (m,n))
  w_arr = np.tile(w_arr, (b, m, 1))
  w = tf.constant(w_arr, shape = (b,m,n))
  w = tf.cast(w, tf.complex64)
  return X * w

def high_on_high_f(X):
  b = 8
  m = 1
  n = 256
  x = np.linspace(-4, 4, num=128)
  weight = sigmoid(x)
  weight = np.append(weight,weight[::-1])
  w_arr = np.reshape(weight, (m,n))
  w_arr = np.tile(w_arr, (b, m, 1))
  w = tf.constant(w_arr, shape = (b,m,n))
  w = tf.cast(w, tf.complex64)
  return X * w
  
def remove_high(X, t):
  b = 8
  m = 1
  n = 256
  x = np.zeros(128)
  x[0:t] = 1
  weight = np.append(x, x[::-1])
  w_arr = np.reshape(weight, (m,n))
  w_arr = np.tile(w_arr, (b, m, 1))
  w = tf.constant(w_arr, shape = (b,m,n))
  w = tf.cast(w, tf.complex64)
  return X * w
  
def fourier_loss(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = tf.fft(yt_radii_cx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = tf.fft(yp_radii_cx)
  #yt_fft = linear_transform(yt_fft)
  #yp_fft = linear_transform(yp_fft)
  #print("\n>>>> yt_fft:")
  #print(yt_fft)
  #print("\n>>>> Absolute value yt_fft:")
  #print(K.abs(yt_fft))
  #print("\n")
  yt_fft = high_on_high_f(yt_fft)
  yp_fft = high_on_high_f(yp_fft)

  loss = K.mean(K.abs(yt_fft - yp_fft)) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
  
def weighted_fourier_loss(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = tf.fft(yt_radii_cx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = tf.fft(yp_radii_cx)

  yt_fft = high_on_low_f(yt_fft)
  yp_fft = high_on_low_f(yp_fft)


  loss = K.mean(K.abs(yt_fft - yp_fft)) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
  
def weighted_fourier_loss2(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = tf.fft(yt_radii_cx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = tf.fft(yp_radii_cx)
  
  yt_fft = remove_high(yt_fft, 64)
  yp_fft = remove_high(yp_fft, 64)

  yt_fft = high_on_low_f2(yt_fft)
  yp_fft = high_on_low_f2(yp_fft)


  loss = K.mean(K.abs(yt_fft - yp_fft)) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
def fourier_loss_less(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = tf.fft(yt_radii_cx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = tf.fft(yp_radii_cx)

  yt_fft = remove_high(yt_fft, 72)
  yp_fft = remove_high(yp_fft, 72)


  loss = K.mean(K.abs(yt_fft - yp_fft)) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
  
def fourier_loss_2less(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = tf.fft(yt_radii_cx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = tf.fft(yp_radii_cx)

  yt_fft = remove_high(yt_fft, 56)
  yp_fft = remove_high(yp_fft, 56)


  loss = K.mean(K.abs(yt_fft - yp_fft)) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
  
def fourier_loss_128(y_true, y_pred):

  yt_radii= K.sum(y_true, axis=2)
  yt_radii_cx = tf.cast(yt_radii, tf.complex64)
  yt_radii_cx = tf.einsum('ijk->ikj', yt_radii_cx)
  yt_fft = K.abs(tf.fft(yt_radii_cx))
  yt_sort = tf.sort(yt_fft, direction='DESCENDING')
  yt_value = yt_sort[0,0,0:15]
  #yt_fft_idx = tf.contrib.framework.argsort(yt_fft, direction='DESCENDING')
  #yt_idx = tf.where(yt_fft > yt_fft[0,0, yt_fft_idx[0,0,128]])
  #yt_value = tf.gather_nd(yt_fft, yt_idx)

  yp_radii= K.sum(y_pred, axis=2)
  yp_radii_cx = tf.cast(yp_radii, tf.complex64)
  yp_radii_cx = tf.einsum('ijk->ikj', yp_radii_cx)
  yp_fft = K.abs(tf.fft(yp_radii_cx))
  yp_sort = tf.sort(yp_fft, direction='DESCENDING')
  yp_value = yp_sort[0,0,0:15]
  #yp_fft_idx = tf.contrib.framework.argsort(yp_fft, direction='DESCENDING')
  #yp_idx = tf.where(yp_fft > yp_fft[0,0, yp_fft_idx[0,0,15]])
  #yp_value = tf.gather_nd(yp_fft, yp_idx)

  loss = K.abs(yt_value - yp_value) # L1 norm
  #loss = K.sum(K.square(yt_fft - yp_fft)) # L2 norm

  return loss
  
def bd_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled) - dice_coef(y_true, y_pred)

def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss

def ce_fourier_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + fourier_loss(y_true, y_pred)

def dice_fourier_loss(y_true, y_pred):
    return 0.01 * fourier_loss(y_true, y_pred) - dice_coef(y_true, y_pred)
    
def dice_fourierless_loss(y_true, y_pred):
    return 0.01 * fourier_loss_less(y_true, y_pred) - dice_coef(y_true, y_pred)
    
def dice_weighted_fourier_loss(y_true, y_pred):
    return 0.01 * weighted_fourier_loss(y_true, y_pred) - dice_coef(y_true, y_pred)
    
# Evaluation metric: IoU
def compute_iou(im1, im2):
    overlap = (im1>0.5) * (im2>0.5)
    union = (im1>0.5) + (im2>0.5)
    return overlap.sum()/float(union.sum())

# Evaluation metric: Dice
def compute_dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1>0.5).astype(np.bool)
    im2 = np.asarray(im2>0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

# Evaluation metric: Sensitivity:
def compute_sensitivity(im1, im2):
    tp = (im1>0.5) * (im2>0.5)
    return tp.sum() / float(im1.sum())
    
# Evaluation metric: Specificity
def compute_specificity(im1, im2):
    tn = (im1<0.5) * (im2<0.5)
    tp = (im1>0.5) * (im2>0.5)
    fp = float(im2.sum()) - float(tp.sum())
    return tn.sum() / (float(tn.sum()) + fp)

########################################
# 2D Standard
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

########################################

"""
Standard U-Net [Ronneberger et.al, 2015]
Total params: 7,759,521
"""
def U_Net(img_rows, img_cols, color_type=1, num_class=1):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(input=img_input, output=unet_output)

    return model

"""
wU-Net for comparison
Total params: 9,282,246
"""
def wU_Net(img_rows, img_cols, color_type=1, num_class=1):

    # nb_filter = [32,64,128,256,512]
    nb_filter = [35,70,140,280,560]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(input=img_input, output=unet_output)

    return model

"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
def UNetPlusPlus(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model


if __name__ == '__main__':
    
    model = U_Net(96,96,1)
    model.summary()

    model = wU_Net(96,96,1)
    model.summary()

    model = UNetPlusPlus(96,96,1)
    model.summary()
