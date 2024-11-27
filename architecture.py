class SiLU(layers.Layer):
    """Custom SiLU/Swish activation layer"""
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        
    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)

class CoordAtt(layers.Layer):
    """Coordinate Attention module"""
    def __init__(self, reduction_ratio=32, **kwargs):
        super(CoordAtt, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.reduced_channels = max(8, channels // self.reduction_ratio)
        
        self.h_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2))
        self.w_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))
        
        self.embedding1 = layers.Dense(
            self.reduced_channels,
            activation='relu',
            kernel_initializer='he_normal'
        )
        
        self.h_embedding2 = layers.Dense(channels, kernel_initializer='he_normal')
        self.w_embedding2 = layers.Dense(channels, kernel_initializer='he_normal')
        
        self.multiply = layers.Multiply()
        self.activation = layers.Activation('sigmoid')

    def call(self, inputs):
        h_att = self.h_pool(inputs)
        w_att = self.w_pool(inputs)
        
        h_att = self.embedding1(h_att)
        w_att = self.embedding1(w_att)
        
        h_att = self.h_embedding2(h_att)
        w_att = self.w_embedding2(w_att)
        
        h_att = tf.expand_dims(h_att, axis=2)
        w_att = tf.expand_dims(w_att, axis=1)
        
        att = self.activation(self.multiply([h_att, w_att]))
        
        return self.multiply([inputs, att])

class ImprovedDualPathBlock(layers.Layer):
    """Enhanced Dual Path Block with attention mechanisms"""
    def __init__(self, dense_channels, residual_channels, proj_channels, 
                 groups=32, stride=1, use_coord_att=True, **kwargs): #dropout_rate=0.2,
        super(ImprovedDualPathBlock, self).__init__(**kwargs)
        self.dense_channels = dense_channels
        self.residual_channels = residual_channels
        self.proj_channels = proj_channels
        self.groups = groups
        self.stride = stride
        self.use_coord_att = use_coord_att
        #self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.proj_channels, 1, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-3)
        )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.act1 = SiLU()
        
        self.conv2 = layers.Conv2D(
            self.proj_channels, 3, strides=self.stride, padding='same',
            groups=self.groups, use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-3)
        )
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.act2 = SiLU()
        
        self.conv3 = layers.Conv2D(
            self.dense_channels + self.residual_channels, 1, padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-3)
        )
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.act3 = SiLU()
        
        if self.use_coord_att:
            self.coord_att = CoordAtt(reduction_ratio=16)
        
        if self.stride > 1 or input_shape[-1] != (self.dense_channels + self.residual_channels):
            self.shortcut = layers.Conv2D(
                self.dense_channels + self.residual_channels, 1,
                strides=self.stride, padding='same', use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(1e-3)
            )
        else:
            self.shortcut = None
            
        self.concat = layers.Concatenate(axis=-1)
        
        self.add = layers.Add()

        # Adding dropout layers
        #self.dropout1 = layers.Dropout(self.dropout_rate)
        #self.dropout2 = layers.Dropout(self.dropout_rate)
        #self.dropout3 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        shortcut = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        #x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        #x = self.dropout2(x, training=training)

        if self.use_coord_att:
            x = self.coord_att(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        #x = self.dropout3(x, training=training)
        
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        
        if shortcut is not None:
            x = layers.Add()([x, shortcut])
        
        dense_features = x[:, :, :, :self.dense_channels]
        residual_features = x[:, :, :, self.dense_channels:]
        
        return self.concat([dense_features, residual_features])

def ImprovedDPN92(input_shape=(224, 224, 3), num_classes=20, include_top=True):
    """Enhanced DPN92 with modern architectural improvements"""

    # Stage configurations
    stages_config = [
        # dense_ch, res_ch, proj_ch, blocks, stride, coord_att
        (256, 16, 96, 3, 1, True),
        (512, 32, 192, 4, 2, True),
        (1024, 24, 384, 20, 2, True),
        (2048, 128, 768, 3, 2, True),
    ]
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Initial conv layers with dropout
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    #x = layers.Dropout(0.2)(x)
    
    # Stem block with early feature mixing
    x = layers.Conv2D(64, 3, padding='same', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = SiLU()(x)
    #x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = SiLU()(x)
    #x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, 3, padding='same', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = SiLU()(x)
    x = layers.MaxPool2D(3, strides=2, padding='same')(x)
    #x = layers.Dropout(0.2)(x)
    
    # Main network stages
    for i, (d_ch, r_ch, p_ch, blocks, stride, use_att) in enumerate(stages_config):
        for b in range(blocks):
            x = ImprovedDualPathBlock(
                dense_channels=d_ch,
                residual_channels=r_ch,
                proj_channels=p_ch,
                stride=stride if b == 0 else 1,
                use_coord_att=use_att,
                #dropout_rate=0.2
            )(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dropout(0.5)(x)
    
    if include_top:
        x = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
    
    return Model(inputs=inputs, outputs=x, name='improved_dpn92')