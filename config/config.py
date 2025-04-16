class Config:
    #detector parameters
    Sacc=3*10**-15
    Sopt=1.5*10**-11
    L=2.5*10**9
    #signal&gap parameters
    zoom_factor=2e-2
    signal_to_gap_length_ratio = 8
    signal_length=1056
    num_samples = 10000
    samp_freq=2
    parameters=[1.1e5,1.2e5,0.9,1,4, 6]#Mtot, M_ratio, R_shift
    scale=5e-5
    signal_length_before_whitened=4096
    #IMR paras
    parameters_IMR=[0.5,1,1e6,2e6,1e6, 2e6]#Deff,m1,m2
    signal_length_IMR=10000
    #physical signal parameters
    f_in=1e-5
    f_out=1
    f_step=1e-5
    #training parameters
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-4
    dropout=0.1
    #model parameters
    hidden_dim= 64
    num_layers = 2
    condition_dim = 3
    #Save
    model_save_path = './saved_models/No_noise/model'
    model_save_path_with_noise = './saved_models/noise/model_with_noise'
    model_save_path_decomposition = './saved_models/mode_decomposition/model_decomposition'
    #Loss
    Loss_coeff=[1e-7,1,1,1]
    Smooth_coeff=[0.001,0.001]
    
    #Position=8
    
    input_dim=1
    output_dim=1
    hidden_dim_E=[8*condition_dim,1028,256,64]#第一个是LSTM的维度
    
    hidden_dim_D=[48,8*condition_dim,64,128,256,64]#第一个是Attention的维度,最后一个是LSTM的维度
    
    #Segmentation
    segment_length=64
    overlap=0.5
    num_token=32
    channels=8
    #Embedding
    EMBEDDING_dim=512
    #CE
    CEkernel_size = (3, 3)
    CEpadding = (1, 1)
    CEout_channels=8
    #ConE
    ConEkernel_size = (3, 3)
    ConEpadding = (1, 1)
    #Transformer
    num_heads = 8
    FF_dim=2048
    num_layers_T=12
    #MLP
    h_dim_MLP=3072

    #Decomposition
    n_modes=4
    Coeff_reconstruction=1