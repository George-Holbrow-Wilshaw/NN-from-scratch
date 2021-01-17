
import numpy as np
import activation_functions as af

class LSTMLayer:

    def __init__(self, sequence, n_units, input_shape, Ct = None, ht_1 = None):
        self.name = 'LSTMLayer'
        self.n_units = n_units
        self.input_size = input_shape
        self.Wf = np.random.rand(self.input_size, n_units)
        self.Wo = np.random.rand(self.input_size, n_units)
        self.Wc = np.random.rand(self.input_size, n_units)
        self.Wi = np.random.rand(self.input_size, n_units)
        self.Wy = np.random.rand(self.input_size, n_units)
        self.bf = np.random.rand(n_units)
        self.bo = np.random.rand(n_units)
        self.bi = np.random.rand(n_units)
        self.bC = np.random.rand(n_units)
        self.by = np.random.rand(n_units)
        self.Ct = np.array([0])
        self.ht = np.array([0])
        self.cell = LSTMcell(Ct, ht_1)

    def forward_pass(self, data):
        
        for timestep in range(0, len(data)):
            # Now we update out cell state and hidden state
            self.Ct, self.ht, self.cell.call(timestep, self.Wf, self.Wo, self.Wc, self.Wi, self.bf, self.bo, self.bi, self.bC, self.Ct, self.ht)
        
        return af.softmax(np.dot(self.Wy, self.ht) + self.by)
    
    def backprop_through_time(self, data):
        
        for timestep in range(0, len(data)):
            # Now we update out cell state and hidden state
            self.Ct, self.ht, self.cell.call(timestep, self.Wf, self.Wo, self.Wc, self.Wi, self.bf, self.bo, self.bi, self.bC, self.Ct, self.ht)
        
        return af.softmax(np.dot(self.Wy, self.ht) + self.by)
 

class LSTMcell:

    def __init__(self, Ct_1, ht_1):
        self.name = 'LSTMLayer'   
        self.Ct_1 = Ct_1
        self.ht_1 = ht_1
        
        # Instantiating gates
        self.input_gate = InputGate()
        self.forget_gate = ForgetGate()
        self.modulation_gate = ModulationGate()
        self.output_gate = OutputGate()


    def call(self, xt, Wf, Wo, Wc, Wi, bf, bo, bi, bC, Ct_1, ht_1):

        ft = self.forget_gate.call(Wf, xt, ht_1, bf)
        it = self.input_gate.call(xt, ht_1, Wi, bi)
        Ct_tilde = self.modulation_gate.call(it, ht_1, xt, Wc, bC)
        Ct = np.multiply(ft, Ct_1) + np.multiply(it, Ct_tilde)
        ht = self.output_gate.call(Ct, ht_1, xt, bo, Wo)
        
        return Ct, ht
         
         
class InputGate(object):

    def call(self, xt, ht_1, Wi, bi):
        concat = np.concatenate(ht_1, xt)
        _it = np.dot(Wi, concat) + bi
        it = af.sigmoid(_it)
        return it

class OutputGate(object):

    def call(self, Ct, ht_1, xt, bo, Wo):
        _ot = np.dot(Wo, np.concatenate(ht_1, xt)) + bo
        ot = af.sigmoid(_ot)
        ht = ot * np.tanh(Ct)
        return ht

class ModulationGate(object):

    def call(self, it, ht_1, xt, Wc, bC): 
        concat = np.concatenate(ht_1, xt)
        _Ct_tilde = np.dot(Wc, concat) + bC
        Ct_tilde = np.tanh(_Ct_tilde)

        itCt_tilde = np.multiply(it, Ct_tilde)
        
        return itCt_tilde

class ForgetGate(object):

    def call(self, Wf, xt, ht_1, bf):
        concat = np.concatenate((xt, ht_1))
        z = np.dot(Wf, concat.T) + bf
        ft = af.sigmoid(z)
        return ft



