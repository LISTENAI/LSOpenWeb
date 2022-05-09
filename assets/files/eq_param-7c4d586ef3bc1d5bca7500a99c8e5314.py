# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:59:34 2021

@author: dwwang3
"""
import scipy.signal as signal
import numpy as np
import math
import pylab as pl
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from scipy import signal

# a_neg_flag = 1

def eq_param(filter_type, N, fs, fc, gain, Q):
    """
    Args:
        fs: Sampling frequency
        fc: Center frequency
        gain: Gain
        Q: Q factor
    """
    
    # common param init
    A = np.sqrt( math.pow(10, (gain/20)) ); 
    w0 = (2 * np.pi * fc) / fs;   
    alpha = math.sin(w0) / (2*Q); 
        
    b0 = 1;
    b1 = 0;
    b2 = 0;
    a0 = 0;
    a1 = 0;
    a2 = 0; 
    
    # print(fc, Q, gain, fs)
    
    if ('LPF' == filter_type):
        #LPF: H(s) = 1 / (s^2 + s/Q + 1)  
        b0 =  (1 - math.cos(w0))/2;
        b1 =   1 - math.cos(w0);
        b2 =  (1 - math.cos(w0))/2;
        a0 =   1 + alpha;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha;

    elif ('HPF' == filter_type):
        #HPF: H(s) = s^2 / (s^2 + s/Q + 1)
        b0 =  (1 + math.cos(w0))/2;
        b1 = -(1 + math.cos(w0));
        b2 =  (1 + math.cos(w0))/2;
        a0 =   1 + alpha;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha;
        
    elif ('BPF' == filter_type):
        #BPF: H(s) = s / (s^2 + s/Q + 1)  (constant skirt gain, peak gain = Q)
        b0 =   math.sin(w0)/2;
        b1 =   0;
        b2 =  -math.sin(w0)/2;
        a0 =   1 + alpha;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha;
        #BPF: H(s) = (s/Q) / (s^2 + s/Q + 1)      (constant 0 dB peak gain)
        # b0 =   alpha;
        # b1 =   0;
        # b2 =  -alpha;
        # a0 =   1 + alpha;
        # a1 =  -2 * math.cos(w0);
        # a2 =   1 - alpha;
        
    elif ('notch' == filter_type):
        #notch: H(s) = (s^2 + 1) / (s^2 + s/Q + 1)
        b0 =   1;
        b1 =  -2 * math.cos(w0);
        b2 =   1;
        a0 =   1 + alpha;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha; 
        
    elif ('APF' == filter_type): 
        #APF: H(s) = (s^2 - s/Q + 1) / (s^2 + s/Q + 1)
        b0 =   1 - alpha;
        b1 =  -2 * math.cos(w0);
        b2 =   1 + alpha;
        a0 =   1 + alpha;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha;
        
    elif ('peaking' == filter_type): 
        #peakingEQ:  H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)
        b0 =   1 + alpha*A;
        b1 =  -2 * math.cos(w0);
        b2 =   1 - alpha*A;
        a0 =   1 + alpha/A;
        a1 =  -2 * math.cos(w0);
        a2 =   1 - alpha/A;
        
    elif ('LowShelf' == filter_type):
        #lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)
        b0 =    A*( (A+1) - (A-1)*math.cos(w0) + 2*math.sqrt(A)*alpha );
        b1 =  2*A*( (A-1) - (A+1)*math.cos(w0)                   );
        b2 =    A*( (A+1) - (A-1)*math.cos(w0) - 2*math.sqrt(A)*alpha );
        a0 =        (A+1) + (A-1)*math.cos(w0) + 2*math.sqrt(A)*alpha;
        a1 =   -2*( (A-1) + (A+1)*math.cos(w0)                   );
        a2 =        (A+1) + (A-1)*math.cos(w0) - 2*math.sqrt(A)*alpha;  
 
    elif ('HighShelf' == filter_type):
        #highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)
        b0 =    A*( (A+1) + (A-1)*math.cos(w0) + 2*math.sqrt(A)*alpha );
        b1 = -2*A*( (A-1) + (A+1)*math.cos(w0)                   );
        b2 =    A*( (A+1) + (A-1)*math.cos(w0) - 2*math.sqrt(A)*alpha );
        a0 =        (A+1) - (A-1)*math.cos(w0) + 2*math.sqrt(A)*alpha;
        a1 =    2*( (A-1) - (A+1)*math.cos(w0)                   );
        a2 =        (A+1) - (A-1)*math.cos(w0) - 2*math.sqrt(A)*alpha; 
      
    else:
        b0 = 1;
        b1 = 0;
        b2 = 0;
        a0 = 1;
        a1 = 0;
        a2 = 0; 
    
    a1 = a1 / a0;
    a2 = a2 / a0;
    b0 = b0 / a0;
    b1 = b1 / a0;
    b2 = b2 / a0;   
    a0 = 1;

    
    # print(a0)
    # print(a1)
    # print(a2)
    # print(b0)
    # print(b1)
    # print(b2)
    
    w, h = signal.freqz([b0,b1,b2], [a0,a1,a2], N);        
    return [b0,b1,b2], [a0,a1,a2], [b0,b1,b2,-a1,-a2], w, h;    
   

def digital_coeffs(f, fs, a, b):
    w = 2 * np.pi * f / fs
    phi = 4 * np.sin(w / 2) ** 2

    # a[1] *= -1
    # a[2] *= -1
    # print(f)
    # print(w)
    # print(phi)
    # print(a)
    # print(b)

    c = 10 * np.log10(
        (b[0] + b[1] + b[2]) ** 2 + (b[0] * b[2] * phi - (b[1] * (b[0] + b[2]) + 4 * b[0] * b[2])) * phi
    ) - 10 * np.log10(
        (a[0] + a[1] + a[2]) ** 2 + (a[0] * a[2] * phi - (a[1] * (a[0] + a[2]) + 4 * a[0] * a[2])) * phi
    )
    return c;



def digital_coeffs2(f, fs, a0, a1, a2, b0, b1, b2):
    w = 2 * np.pi * np.array(f) / fs
    phi = 4 * np.sin(w / 2) ** 2

    # a1 *= -1
    # a2 *= -1

    c = 10 * np.log10(
        (b0 + b1 + b2) ** 2 + (b0 * b2 * phi - (b1 * (b0 + b2) + 4 * b0 * b2)) * phi
    ) - 10 * np.log10(
        (a0 + a1 + a2) ** 2 + (a0 * a2 * phi - (a1 * (a0 + a2) + 4 * a0 * a2)) * phi
    )
    return c

def response_plot(name, X, H):
    fig, ax = pl.subplots();
    plt.ylim(-15, 20);
    # pl.semilogx();
    pl.xlim([0, 8000])
    pl.grid(True, which='major')
    pl.grid(True, which='minor')
    ax.set_ylabel('Amplitude [dB]', color='b');
    ax.set_xlabel('Frequency');
    ax.set_title(name + ' Digital filter frequency response');
    ax.plot(X, 20 * np.log10(abs(H)), 'g');

def log_file(h):
    fo = open('eq_param.h', "w")
    fo.write("int32_t eq_param[] = \n")
    fo.write("{\n")
    for i in range(len(h)):
        real_int = int(h[i].real * (2**15 - 1));
        # print(i, ',', real_int);
        fo.write(str(real_int) + ", ");
        imag_int = int(h[i].imag * (2**15 - 1));
        # print(i, ',', imag_int);
        fo.write(str(imag_int) + ", ");
        
        fo.write("\n");
        if 0 == (i+1)%16:
            fo.write("\n");
    fo.write("};")
    fo.close()
    
    
def log_param_ba(param_ba):
    fo = open('eq_param_ba.h', "w")
    fo.write("eq_param[] = /* b0, b1, b2, -a1, -a2 */ \n")
    fo.write("{\n")
    
    param = np.array(param_ba).astype(np.float32)
    for i in range(len(param)):
        data =  param[i];       
        fo.write("\t" + str(data) + ",\t");
        if 0 == (i+1)%5:
            fo.write("\n");
            
    fo.write("};")
    fo.close()
    
    
def RCV_PATH():
    N  = 256;
    FS = 48000;
    X = np.arange(N)
    for i in range(0, N):
        X[i] = i*((FS/2)/N);

    param_ba = []
     
    # 合肥实验室配置
    # b, a, ba, w, h0= eq_param('peaking', N, FS, 300, -3, 1);        param_ba += ba; 
    # b, a, ba, w, h1 = eq_param('peaking', N, FS, 3500, 6, 2);       param_ba += ba; 
    # b, a, ba, w, h2 = eq_param('peaking', N, FS, 2000, -10, 2);      param_ba += ba; 
    # b, a, ba, w, h3 = eq_param('peaking', N, FS, 1000, 0.5, 1);      param_ba += ba;
    # b, a, ba, w, h4 = eq_param('peaking', N, FS, 230,  0, 3);        param_ba += ba; 
    # b, a, ba, w, h5 = eq_param('peaking', N, FS, 900,  2, 1);        param_ba += ba; 
    # h = h0 * h1 * h2 * h3 * h4 * h5
    #东莞zoom认证版本

    # b, a, ba, w, h0 = eq_param('HighShelf', N, FS,  6000, 25, 0.8);       param_ba += ba; 
    # b, a, ba, w, h1 = eq_param('peaking',   N, FS,  200,  4,  0.6);       param_ba += ba; 
    # b, a, ba, w, h2 = eq_param('peaking',   N, FS,  2000, -11,  0.9);     param_ba += ba; 
    # b, a, ba, w, h3 = eq_param('peaking',   N, FS,  500,  -2,  0.6);      param_ba += ba; 
    # b, a, ba, w, h4 = eq_param('peaking',   N, FS,  4000,  8,   4);       param_ba += ba; 
    # b, a, ba, w, h5 = eq_param('peaking',   N, FS,  5000,  -6,  4);       param_ba += ba; 
    # b, a, ba, w, h6 = eq_param('peaking',   N, FS,  6000,  3,   1);       param_ba += ba;
    
    #EMEETING
    #b, a, ba, w, h0 = eq_param('HPF', N, FS,  10, 0.1, 0.8);       param_ba += ba; 
    #b, a, ba, w, h1 = eq_param('peaking', N, FS,  200, -50, 1);       param_ba += ba; 
    b, a, ba, w, h0 = eq_param('HPF', N, FS,  300, 0.1, 0.8);       param_ba += ba; 
      
    # h = h0 * h1 * h2 * h3 * h4 * h5  *h6
    h = h0
    response_plot('', X, h);
    log_file(h)
    log_param_ba(param_ba)
    
    
def SEND_PATH():
    N  = 256;
    FS = 16000;
    X = np.arange(N)
    for i in range(0, N):
        X[i] = i*((FS/2)/N);

    param_ba = [] 
    #zoom 认证
    '''
    b, a, ba, w, h0 = eq_param('peaking',  N, FS,  2000, 2, 5);       param_ba += ba; 
    b, a, ba, w, h1 = eq_param('peaking',  N, FS,  4000, -3, 0.9);    param_ba += ba; 
    b, a, ba, w, h2 = eq_param('peaking',   N, FS, 3000, -3, 2);      param_ba += ba; 
    b, a, ba, w, h3 = eq_param('HighShelf', N, FS, 7000, 10, 1);    param_ba += ba; 
    '''
    
    ''' 
    苏研院
    b, a, ba, w, h0 = eq_param('peaking',  N, FS,  2500, -5, 1);       param_ba += ba; 
    b, a, ba, w, h1 = eq_param('peaking',  N, FS,  7500, 3, 1);    param_ba += ba; 
    b, a, ba, w, h2 = eq_param('peaking',  N, FS,  200, -2, 2);    param_ba += ba; 
    #b, a, ba, w, h2 = eq_param('peaking',   N, FS, 3000, -3, 2);      param_ba += ba; 
    #b, a, ba, w, h3 = eq_param('HighShelf', N, FS, 6000, 30, 0.8);    param_ba += ba;
    #b, a, ba, w, h4 = eq_param('peaking', N, FS, 600, -3, 1.5);    param_ba += ba;
    '''
    '''
    b, a, ba, w, h0 = eq_param('peaking',  N, FS,  3800, -3, 3);       param_ba += ba; 
    b, a, ba, w, h1 = eq_param('peaking',  N, FS,  5500, 3, 5);       param_ba += ba; 
    b, a, ba, w, h2 = eq_param('peaking',  N, FS,  900, 4, 0.5);       param_ba += ba;
    b, a, ba, w, h3 = eq_param('HighShelf', N, FS, 7000, 10, 1);    param_ba += ba;  
    
    b, a, ba, w, h4 = eq_param('peaking',  N, FS,  220, 3, 1);       param_ba += ba; 
    b, a, ba, w, h5 = eq_param('peaking',  N, FS,  320, 3, 1);       param_ba += ba; 
    '''
    
    
    '''
    b, a, ba, w, h0 = eq_param('peaking',  N, FS,  3500, -6, 2);      param_ba += ba; 
    b, a, ba, w, h1 = eq_param('peaking',  N, FS,  5200, 6, 8);       param_ba += ba; 
    b, a, ba, w, h2 = eq_param('peaking',  N, FS,  900, 6, 3);      param_ba += ba;
    b, a, ba, w, h3 = eq_param('HighShelf', N, FS, 7000, 10, 1);      param_ba += ba;
    
    b, a, ba, w, h4 = eq_param('peaking',  N, FS,  220, 12, 4);        param_ba += ba;
    b, a, ba, w, h5 = eq_param('peaking',  N, FS,  320, 9, 4);        param_ba += ba;
    b, a, ba, w, h6 = eq_param('peaking',  N, FS,  1460, -2, 4);      param_ba += ba;
    '''
      
    
    
    b, a, ba, w, h0 = eq_param('peaking',  N, FS,  220, 12, 4);        param_ba += ba;
    b, a, ba, w, h1 = eq_param('peaking',  N, FS,  320, 9, 4);        param_ba += ba;

    b, a, ba, w, h2 = eq_param('peaking',  N, FS,  900, 6, 3);      param_ba += ba;
    b, a, ba, w, h3 = eq_param('peaking',  N, FS,  1460, -2, 4);      param_ba += ba;
  
    b, a, ba, w, h4 = eq_param('peaking',  N, FS,  3400, -6, 3);      param_ba += ba; 
    b, a, ba, w, h5 = eq_param('peaking',  N, FS,  5450, 6, 4);       param_ba += ba; 
    b, a, ba, w, h6 = eq_param('HighShelf', N, FS, 6500, 15, 1);      param_ba += ba;  
    

    h = h0 * h1 * h2 * h3 * h4 * h5 * h6

    response_plot('', X, h);
    log_file(h)
    log_param_ba(param_ba)
    
# RCV PATH
if __name__ == '__main__':
    RCV_PATH() 
    #SEND_PATH()

    
    