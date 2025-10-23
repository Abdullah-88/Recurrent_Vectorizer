import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class VectorDynamicTanh(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.alpha = nn.Parameter(torch.randn(input_shape))
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x



class HyperVectorDynamicTanh(nn.Module):
    def __init__(self):
    
        super().__init__()
       
           
    def forward(self, x, alpha):
        x = torch.tanh(alpha * x)
        return x


      
class GRUCell(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.input_size = dim
        self.hidden_size = dim
        self.reset_parameters()
        self.x_reset_vdyt = VectorDynamicTanh(dim)
        self.x_upd_vdyt = VectorDynamicTanh(dim)
        self.x_new_vdyt = VectorDynamicTanh(dim)
        self.h_reset_vdyt = VectorDynamicTanh(dim)
        self.h_upd_vdyt = VectorDynamicTanh(dim)
        self.h_new_vdyt = VectorDynamicTanh(dim)

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h=None):

       

        if h is None:
            h = Variable(x.new_zeros(input.size(0), self.hidden_size))

        


        x_reset = self.x_reset_vdyt(x)
        x_upd = self.x_upd_vdyt(x)
        x_new = self.x_new_vdyt(x)
        h_reset = self.h_reset_vdyt(h)
        h_upd = self.h_upd_vdyt(h)
        h_new = self.h_new_vdyt(h)

        reset_gate = torch.tanh(x_reset + h_reset)
        update_gate = torch.tanh(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * h + (1 - update_gate) * new_gate

        return hy
                    	   
class GRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
      
        
        self.layer_dim = dim
        self.hidden_dim = dim
        self.gru_cell = GRUCell(dim)
              
             
    def forward(self, x):
        
      
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
         
       
        outs = []
        
        hn = h0[0,:,:]
        
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)
            

        
        out = torch.stack(outs, dim=1)
        
        return out 

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.token_vdyt = VectorDynamicTanh(dim)
      
      
             	   
    def forward(self, x):
    
        x = self.token_vdyt(x)    	
      
        
        return x
    	

class GlobalMappingUnit(nn.Module):
    def __init__(self,dim,num_tokens):
        super().__init__()
        
             
        self.state_vdyt = VectorDynamicTanh(dim)
        self.probe_vdyt = VectorDynamicTanh(dim) 
        self.gru = GRU(dim)       
        self.readout_hvdyt = HyperVectorDynamicTanh()
              
                                      	   
    def forward(self, x):
    
       
        state = self.state_vdyt(x)
        probe = self.probe_vdyt(x)
        alpha = self.gru(state)
        readout = self.readout_hvdyt(probe, alpha)
        
        return readout          




class VectorizerBlock(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model, num_tokens)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class Vectorizer(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[VectorizerBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








