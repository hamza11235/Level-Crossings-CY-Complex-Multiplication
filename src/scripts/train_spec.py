from utils_spec import*



def train_nn(psi, dim_amb=3, epochs=5):
    
    """
    Learning a Ricci-flat metric for the Quartic K3 surface 
    """
    
    file_dir = "quartic_{:}".format(psi)
    data = np.load(os.path.join(file_dir, 'dataset.npz'))
    BASIS = np.load(os.path.join(file_dir, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_tf_basis(BASIS)
    
    n_out = 1
    n_in = 2*(dim_amb + 1)
    nlayer=3
    act='gelu'
    nHidden=64
    alpha = [1., 1., 1., 1., 1.]

    nn_phi = tf.keras.Sequential()
    nn_phi.add(tfk.Input(shape=(n_in)))
    for i in range(nlayer):
        nn_phi.add(tfk.layers.Dense(nHidden, activation=act))
    nn_phi.add(tfk.layers.Dense(n_out, use_bias=False))

    scb = SigmaCallback((data['X_val'], data['y_val']))
    volkcb = VolkCallback((data['X_val'], data['y_val']))
    cb_list = [scb, volkcb]

    phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
    cmetrics = [TotalLoss(), SigmaLoss(), VolkLoss()]
    opt_phi = tfk.optimizers.Adam()
    phimodel, training_history = train_model(phimodel, data, optimizer=opt_phi, epochs=5, batch_sizes=[64, 10000], 
                                           verbose=1, custom_metrics=cmetrics, callbacks=cb_list)
    phimodel.save(file_dir)
    return phimodel, training_history
