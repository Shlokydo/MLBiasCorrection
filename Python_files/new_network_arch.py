import numpy as np
import tensorflow as tf

#Model network defination
class rnn_model(tf.keras.Model):

    def __init__(self, parameter_list, name = 'RNN_Model'):
        super(rnn_model, self).__init__()
        self.unit = parameter_list['LSTM_output']
        self.acti = parameter_list['activation']
        self.recurrent_activ = parameter_list['rec_activation']
        self.kernel_regular = tf.keras.regularizers.l2(parameter_list['l2_regu'])
        self.activity_regular = tf.keras.regularizers.l1(parameter_list['l1_regu'])
        self.drop = parameter_list['lstm_dropout']
        self.recurrent_drop = parameter_list['rec_lstm_dropout']
        self.unro = parameter_list['unroll_lstm']
        self.return_sequence = True
        self.statef = parameter_list['stateful']
        self.num_layers = parameter_list['num_lstm_layers']
        self.num_dense_layers = parameter_list['num_dense_layers']
        self.dense_out = parameter_list['dense_output']
        self.net_out = parameter_list['net_output']
        self.locality = parameter_list['locality']

    def build(self, input_shape):

        self.lstm_seq = tf.keras.Sequential()

        for i in range(self.num_layers):

            self.lstm_seq.add(tf.keras.layers.LSTM(units = self.unit[i], 
                                                    activation = self.acti,
                                                    recurrent_activation = self.recurrent_activ,
                                                    kernel_regularizer = self.kernel_regular,
                                                    activity_regularizer = self.activity_regular,
                                                    dropout = self.drop,
                                                    recurrent_dropout = self.recurrent_drop,
                                                    unroll = self.unro,
                                                    return_sequences = self.return_sequence,
                                                    stateful = self.statef,
                                                    name = 'LSTM_{}'.format(i+1)))
        
        for i in range(self.num_dense_layers - 1):

            self.lstm_seq.add(tf.keras.layers.Dense(units=self.dense_out[i],
                                    kernel_regularizer = self.kernel_regular,
                                    activation = None,
                                    name = 'DENSE_{}'.format(i+1)))
            self.lstm_seq.add(tf.keras.layers.ELU(1.5, name='ELU_{}'.format(i+1)))

        self.lstm_seq.add(tf.keras.layers.Dense(units = self.net_out,
                                    kernel_regularizer = self.kernel_regular,
                                    activation = None,
                                    name = 'DENSE_OUTPUT'))

        self.lstm_seq.add(tf.keras.layers.ELU(1.5, name = 'ELU_output'))

    def call(self, inputs):

        return tf.expand_dims(inputs[:,:,int(self.locality/2)], axis=1) + self.lstm_seq(inputs)