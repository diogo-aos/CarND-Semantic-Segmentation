import tensorflow as tf

USE_SCALING = False
USE_REGULIZER = False

def load_vgg(sess, vgg_path):
    """                                                                                                                                                                 
    Load Pretrained VGG Model into TensorFlow.                                                                                                                          
    :param sess: TensorFlow Session                                                                                                                                     
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"                                                                                   
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)                                                               
    """
    # TODO: Implement function                                                                                                                                          
    #   Use tf.saved_model.loader.load to load the model and weights                                                                                                    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'                                                                                                                         
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    g = tf.get_default_graph()
    input_layer = g.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = g.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = g.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = g.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = g.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_layer, keep_prob, l3, l4, l7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """                                                                                                                                                                 
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.                                                                       
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output                                                                                                             
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output                                                                                                             
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output                                                                                                             
    :param num_classes: Number of classes to classify                                                                                                                   
    :return: The Tensor for the last layer of output                                                                                                                    
    """
    # at the end of VGG16, the image has been downsampled to 1/32th of original size
    # Part1 (upsample 2x, final=2x):
    #   conv 1x1 of layer7
    #   upsample 2x previous
    #   conv 1x1 layer4
    #   add them
    # Part2 (upsample 2x, final=4x):
    #   upsample 2x Part1
    #   conv 1x1 layer3
    #   add them
    # Part3 (upsample 8x, final=32x):
    #   upsample Part2 8x
    
    # scaling layers
    if USE_SCALING:
        vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
        vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')
    
    # TODO: Implement function
    # 1 by 1 convolution from VGG output
    # stride is what is upsmapling, padding must be same, size may change
    # use regulizer to everylayer according to Aron
    # the regulizer penalizes when the weights get too large
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                filters=num_classes,
                                kernel_size=1,
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #fcn32 = tf.layers.conv2d_transpose(l7_conv_1x1,
    #                                    filters=num_classes,
    #                                    kernel_size=64,
    #                                    strides=32,
    #                                    padding='same',
    #                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out,
                                filters=num_classes,
                                kernel_size=1,
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out,
                                filters=num_classes,
                                kernel_size=1,
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # second layer is transposed convolution from 1 by 1 convolution
    # we want to upsample x2 and then skip layer 4
    fcn16 = tf.layers.conv2d_transpose(l7_conv_1x1,
                                        filters=num_classes,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    fcn16 = tf.add(fcn16, l4_conv_1x1)
    #fcn16 = tf.layers.conv2d_transpose(fcn16,
    #                                    filters=num_classes,
    #                                    kernel_size=32,
    #                                    strides=16,
    #                                    padding='same',
    #                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # next we want to again upsample x2 (total 4x),
    # skip upsampled layer 4 (upsample x2)
    # skip layer 3
    fcn8 = tf.layers.conv2d_transpose(fcn16,
                                        filters=num_classes,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    fcn8 = tf.add(fcn8, l3_conv_1x1)
    fcn8 = tf.layers.conv2d_transpose(fcn8,
                                        filters=num_classes,
                                        kernel_size=16,
                                        strides=8,
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return fcn8


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """                                                                                                                                                                 
    Build the TensorFLow loss and optimizer operations.                                                                                                                 
    :param nn_last_layer: TF Tensor of the last layer in the neural network                                                                                             
    :param correct_label: TF Placeholder for the correct label image                                                                                                    
    :param learning_rate: TF Placeholder for the learning rate                                                                                                          
    :param num_classes: Number of classes to classify                                                                                                                   
    :return: Tuple of (logits, train_op, cross_entropy_loss)                                                                                                            
    """
    # logits : 2D tensor; rows=pixels; columns=pixel classes
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label, name='cross_entropy')
    loss_operation = tf.reduce_mean(cross_entropy, name='loss')

    if USE_REGULIZER:
        reg_constant = 1
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_operation = loss_operation + reg_constant * sum(reg_losses)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_operation)
    return logits, train_op, loss_operation
