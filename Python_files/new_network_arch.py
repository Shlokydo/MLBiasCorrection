import numpy as np
import tensorflow as tf

#Model network defination
def rnn_model(tf.keras.Model):

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
    self.num_layers = parameter_list['num_lstm_layers']

  def build(self, input_shape):

    self.lstm_seq = tf.keras.Sequential()

    for i in range(self.num_layers):

      self.lstm_seq.add(tf.keras.layers.LSTM(units = self.unit, 
                                              activation = self.acti,
                                              recurrent_activation = self.recurrent_acti,
                                              kernel_regularizer = self.kernel_regular,
                                              ))

