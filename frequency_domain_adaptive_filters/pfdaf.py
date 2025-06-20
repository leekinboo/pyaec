# Copyright 2020 ewan xu<ewan_xu@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

""" Partitioned-Block-Based Frequency Domain Adaptive Filter """

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft

class PFDAF:
  def __init__(self, N, winlen, mu, partial_constrain):
    self.N = N
    self.M = winlen
    self.R = np.zeros((self.N,self.N_freq),dtype=np.float32)
    self.D = np.zeros((self.N,self.N_freq),dtype=np.float32)
    self.C = np.zeros((self.N,self.N_freq),dtype=np.float32)

    self.N_freq = 1+winlen
    self.N_fft = 2*winlen
    self.mu = mu
    self.partial_constrain = partial_constrain
    self.p = 0
    self.x_old = np.zeros(self.M,dtype=np.float32)
    self.X = np.zeros((N,self.N_freq),dtype=np.complex)
    self.H = np.zeros((self.N,self.N_freq),dtype=np.complex)
    self.window = np.hanning(self.M)

import numpy as np

def conjugate_gradient_update(obj, ibin, i_comp, c_buf_Y_rx, lambda_val):
    """
    执行共轭梯度算法的单次迭代更新（向量化版本）
    
    参数:
    obj: 包含算法状态的字典
    ibin: 频点索引（向量或标量）
    i_comp: 自相关矩阵索引
    c_buf_Y_rx: 误差相关系数
    lambda_val: 遗忘因子
    
    返回:
    obj: 更新后的算法状态
    """
    # 计算梯度向量
    R_toe = obj['Rtoe'][ibin, i_comp]
    toeplitz_R = np.array([np.diagflat(np.ones(N - np.abs(k))) * R_toe[k] 
                          for k in range(-N+1, N)]).sum(axis=0)
    
    obj['g0'][ibin] = obj['rcross'][ibin] - (toeplitz_R @ obj['w_last'][ibin].T).T
    
    # 初始化搜索方向
    obj['p'][ibin] = obj['g0'][ibin]
    obj['gd_last'][ibin] = obj['g0'][ibin]
    
    # 计算误差相关项
    obj['v'][ibin] = c_buf_Y_rx * obj['e'][ibin]
    
    # 计算R(k)p(k)
    obj['rp'][ibin] = (toeplitz_R @ obj['p'][ibin].T).T
    
    # 步长计算
    numerator = (obj['p'][ibin].conj() @ obj['gd_last'][ibin].T).reshape(-1, 1)
    denominator = (obj['p'][ibin].conj() @ obj['rp'][ibin].T).reshape(-1, 1) + 1e-30
    obj['alf'][ibin] = lambda_val * numerator / denominator
    
    # 权重更新
    obj['w'][ibin] = obj['w_last'][ibin] + obj['alf'][ibin] * obj['p'][ibin]
    
    # 参数β计算
    g_current = obj['g0'][ibin]
    g_prev = obj['gd_last'][ibin]
    g_current_norm_sq = (g_current.conj() @ g_current.T).reshape(-1, 1)
    g_prev_norm_sq = (g_prev.conj() @ g_prev.T).reshape(-1, 1) + 1e-30
    obj['beta'][ibin] = g_current_norm_sq / g_prev_norm_sq
    
    # 更新搜索方向
    obj['p_next'][ibin] = g_current + obj['beta'][ibin] * obj['p'][ibin]
    
    # 更新梯度历史
    obj['gd_last'][ibin] = g_current
    obj['p'][ibin] = obj['p_next'][ibin]
    
    return obj
  
  def filt(self, x, d):
    assert(len(x) == self.M)
    x_now = np.concatenate((self.x_old, x))
    d_now = np.concatenate((self.d_old, d))

    X = fft(x_now)
    D = fft(d_now)
    self.X[1:] = self.X[:-1]
    self.D[1:] = self.D[:-1]

    self.X[0] = X
    self.D[0] = D
    
    self.Rnew = np.dot(self.X[1:],np.conj*(self.X[1:])) #to do , 
    self.R = (self.R * (self.N-1) + self.Rnew) / self.N 
    
    self.x_old = x
    self.d_old = d
    
    self.Cnew = np.dot(self.X[1:],D) #to do , 
    
    self.C = (self.C*(self.N-1) + self.Cnew) / self.N 

    Y = np.sum(self.H*self.X,axis=0)

    y = ifft(Y)[self.M:]
    e = d-y
  

    toeplitz_R = np.array([np.diagflat(np.ones(N - np.abs(k))) * R_toe[k] 
                          for k in range(-N+1, N)]).sum(axis=0)
    
    g = self.C - (toeplitz_R @ self.H)
    
    p = g0
    gd_last= g0
    
    # 计算误差相关项
    v = c_buf_Y_rx * e
    
    # 计算R(k)p(k)
  
    rp = (toeplitz_R @ p).T
    
    # 步长计算
    numerator = (p.conj() @ gd_last.T)
    denominator =  (p.conj() @ rp.T) + 1e-30;
    
    alpha = lambda_val * numerator / denominator
    
    # 权重更新
    w = w_last + alpha * p 
    
    # 参数β计算
    g_current = obj['g0'][ibin]
    g_prev = obj['gd_last'][ibin]
    g_current_norm_sq = (g_current.conj() @ g_current.T).reshape(-1, 1)
    g_prev_norm_sq = (g_prev.conj() @ g_prev.T).reshape(-1, 1) + 1e-30
    obj['beta'][ibin] = g_current_norm_sq / g_prev_norm_sq
    
    # 更新搜索方向
    obj['p_next'][ibin] = g_current + obj['beta'][ibin] * obj['p'][ibin]
    
    # 更新梯度历史
    obj['gd_last'][ibin] = g_current
    obj['p'][ibin] = obj['p_next'][ibin]
    
  
  def update(self,e):
    X2 = np.sum(np.abs(self.X)**2,axis=0)
    e_fft = np.zeros(shape=(self.N_fft,),dtype=np.float32)
    e_fft[self.M:] = e*self.window
    E = fft(e_fft)
    
    G = self.mu*E/(X2+1e-10)
    self.H += self.X.conj()*G

    if self.partial_constrain:
      h = ifft(self.H[self.p])
      h[self.M:] = 0
      self.H[self.p] = fft(h)
      self.p = (self.p + 1) % self.N
    else:
      for p in range(self.N):
        h = ifft(self.H[p])
        h[self.M:] = 0
        self.H[p] = fft(h)
  

                            
 def pfdaf(x, d, N=4, M=64, mu=0.2, partial_constrain=True):
    ft = PFDAF(N, M, mu, partial_constrain)
    num_block = min(len(x),len(d)) // M

  e = np.zeros(num_block*M)
  for n in range(num_block):
    x_n = x[n*M:(n+1)*M]
    d_n = d[n*M:(n+1)*M]
    e_n = ft.filt(x_n,d_n)
    ft.update(e_n)
    e[n*M:(n+1)*M] = e_n
    
  return e
