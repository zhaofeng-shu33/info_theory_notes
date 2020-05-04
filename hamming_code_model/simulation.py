#!/usr/bin/python
# -*- coding: utf-8 -*-
# author : zhaofeng-shu33
# file-description : use hamming(3,1), hamming(7,4), hamming(15,11) 
# three different coding scheme to investigate how the error-probability changes
# as the transfer probability of BSC changes.
# license : Apache License Version 2.0

import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import logging
import pdb
class Hamming_BASE:
    def __init__(self, generator_matrix, parity_check_matrix, decoding_matrix):
        self.G = generator_matrix
        self.H = parity_check_matrix
        self.R = decoding_matrix
        self.message_length = self.G.shape[1]
        self.block_length = self.G.shape[0]
        self.error_check_array = np.logspace(0,self.block_length - self.message_length - 1, self.block_length - self.message_length, base = 2, dtype = int)
        self.epsilon_list = np.array([0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50])
    def encode(self, data):
        reshaped_data = data.reshape([int(data.size/self.message_length),self.message_length]).T
        encoded_reshaped_data = np.mod(self.G @ reshaped_data, 2)
        return encoded_reshaped_data.T.reshape(encoded_reshaped_data.size)
    def decode(self, data):
        data_after_error_checked = self._error_check(data)        
        assert(np.mod(data.size, self.block_length) == 0)
        reshaped_data = data_after_error_checked.reshape([int(data.size/self.block_length), self.block_length]).T        
        decoded_reshaped_data = np.mod(self.R @ reshaped_data, 2)
        return decoded_reshaped_data.T.reshape(decoded_reshaped_data.size)
    def _error_check(self, data):
        index = 0
        data_after_error_checked = data.copy()
        while(index < data.size):
            indicator = np.dot(self.error_check_array, np.mod(self.H @ data[index:(index + self.block_length)],2))
            if not(indicator == 0): # correct it
                data_after_error_checked[index + indicator - 1] = 1- data[index + indicator - 1]
            index += self.block_length
        return data_after_error_checked
    def bit_error_rate_vs_BSC_error(self, input_data):
        input_data_encoded = self.encode(input_data)
        self.bit_error_list = np.zeros(self.epsilon_list.__len__())
        for index, epsilon in enumerate(self.epsilon_list):
            received_data = bsc_transfer(input_data_encoded, error_rate = epsilon)
            decoded_data = self.decode(received_data)
            # calculate the error rate
            error_rate = sum(input_data != decoded_data) * 1.0 /input_data.size
            logging.info('for BSC e = %f, bit error rate = %f'%(epsilon, error_rate))
            self.bit_error_list[index] = error_rate
                
class Hamming_7_4(Hamming_BASE):
    def __init__(self):
        # construct the generator matrix
        G_7_4 = np.asarray([1, 1, 0, 1,
                            1, 0, 1, 1,
                            1, 0, 0, 0,
                            0, 1, 1, 1,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1], dtype = bool).reshape([7,4])
        H_7_4 = np.array([1, 0, 1, 0, 1, 0, 1,
                        0, 1, 1, 0, 0, 1, 1,
                        0, 0, 0, 1, 1, 1, 1], dtype = bool).reshape([3, 7])
        R_7_4 = np.array([0, 0, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 1], dtype = bool).reshape(4, 7)
        super(Hamming_7_4, self).__init__(G_7_4, H_7_4, R_7_4)
    def theorical_bound(self):
        beta = 2 * np.sqrt(self.epsilon_list * (1 - self.epsilon_list))
        return 7 * np.power(beta, 3) + 7 * np.power(beta, 4) + np.power(beta, 7)
class Hamming_3_1(Hamming_BASE):
    def __init__(self):
        G_3_1 = np.array([1,1,1], dtype = bool).reshape(3,1)
        H_3_1 = np.array([1, 0, 1, 0, 1, 1], dtype = bool).reshape(2, 3)
        R_3_1 = np.array([0, 0, 1], dtype = bool).reshape(1, 3)
        super(Hamming_3_1, self).__init__(G_3_1, H_3_1, R_3_1)
class Hamming_15_11(Hamming_BASE):
    def __init__(self):
        # construct the generator matrix
        G_15_11 = np.asarray([1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                            1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,                            
                            ], dtype = bool).reshape([15,11])
        H_15_11 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                          0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                          0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype = bool).reshape([4, 15])
        R_15_11 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1                          
                          ], dtype = bool).reshape(11, 15)
        super(Hamming_15_11,self).__init__(G_15_11, H_15_11, R_15_11)

def bsc_transfer(data, error_rate):
    '''
        transfer data through this noisy channel
    '''
    data_noise = bernoulli.rvs(p = error_rate, size = data.size)
    index_corrupted = np.where(data_noise == 1)
    data_output = data.copy()
    data_output[index_corrupted] = 1 - data[index_corrupted]
    return data_output

def generate(num_of_data):
    return bernoulli.rvs(p = 0.5, size = num_of_data)


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    # class object initialization
    H_3_1_instance = Hamming_3_1()
    H_7_4_instance = Hamming_7_4()
    H_15_11_instance = Hamming_15_11()    
    # pdb.set_trace()    
    input_data = generate(3300)
    H_7_4_instance.bit_error_rate_vs_BSC_error(input_data)
    H_15_11_instance.bit_error_rate_vs_BSC_error(input_data)
    H_3_1_instance.bit_error_rate_vs_BSC_error(input_data)    
    plt.plot(H_7_4_instance.epsilon_list, H_7_4_instance.bit_error_list, 'r', label = 'H(7,4)')
    plt.plot(H_15_11_instance.epsilon_list, H_15_11_instance.bit_error_list, 'b', label = 'H(15,11)')    
    plt.plot(H_3_1_instance.epsilon_list, H_3_1_instance.bit_error_list, 'g', label = 'H(3,1)')        
    plt.title('bit error rate versus BSC error probability')
    plt.legend()
    plt.savefig('hamming.eps')
    plt.show()
