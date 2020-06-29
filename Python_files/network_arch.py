import numpy as np
import tensorflow as tf

tf.random.set_seed(5)

def cus_tanh(x):
    return 2 * tf.keras.activations.tanh(x)

#Model network defination
class rnn_model(tf.keras.Model):

    def __init__(self, parameter_list, name = 'RNN_Model'):
        super(rnn_model, self).__init__()
        self.unit = parameter_list['LSTM_output']
        self.acti = parameter_list['activation']
        self.acti_d = parameter_list['d_activation']
        self.recurrent_activ = parameter_list['rec_activation']
        self.kernel_regular = tf.keras.regularizers.l2(parameter_list['l2_regu'])
        self.activity_regular = tf.keras.regularizers.l1(parameter_list['l1_regu'])
        self.drop = parameter_list['lstm_dropout']
        self.recurrent_drop = parameter_list['rec_lstm_dropout']
        self.num_layers = parameter_list['num_lstm_layers']
        self.num_dense_layers = parameter_list['num_dense_layers']
        self.dense_out = parameter_list['dense_output']
        self.locality = parameter_list['locality']
        self.dense_drop = parameter_list['dense_drop']

    def build(self, input_shape):

        self.lstm_list = []
        i = 0
        for i in range(self.num_layers):

            self.lstm_list.append(tf.keras.layers.LSTM(units = self.unit[i], 
                                                    activation = self.acti,
                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    dropout = self.drop,
                                                    recurrent_dropout = self.recurrent_drop,
                                                    return_sequences = True,
                                                    name = 'LSTM_{}'.format(i+1), 
                                                    return_state=True))

        self.dense_list = []
        i = 0
        for i in range(self.num_dense_layers):
            self.dense_list.append(tf.keras.layers.Dense(units=self.dense_out[i],
                                    kernel_regularizer = self.kernel_regular,
                                    activation = self.acti_d,
                                    name = 'DENSE_{}'.format(i+1)))
            self.dense_list.append(tf.nn.dropout(self.dense_drop))
        self.dense_list.append(tf.keras.layers.Dense(units=self.dense_out[-1],
                                            kernel_regularizer = self.kernel_regular,
                                            activation = None,
                                            name = 'DENSE_{}'.format(i+1)))
    def call(self, inputs, stat):
        
        initializer = tf.initializers.GlorotUniform(seed = 1)
        state_h = [initializer([inputs.shape[0], self.unit[i]], dtype = tf.dtypes.float32) for i in range(self.num_layers)]
        state_c = [initializer([inputs.shape[0], self.unit[i]], dtype = tf.dtypes.float32) for i in range(self.num_layers)]
        x = inputs
        for i in range(len(self.lstm_list)):
            try:
                x, state_h[i], state_c[i] = self.lstm_list[i](x, initial_state = [stat[0][i], stat[1][i]], training = True)
            except:
                x, state_h[i], state_c[i] = self.lstm_list[i](x, initial_state = [state_h[i], state_c[i]], training = True)
        
        #Only using last time-step as the input to the dense layer
        try:
            x = x[:, -1, :]
        except:
            pass
        for i in range(len(self.dense_list)):
            x = self.dense_list[i](x)
        
        return x, [state_h, state_c]
