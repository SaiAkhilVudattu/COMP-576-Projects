class configration():

    def __init__(self):
        self.nn_input_dim = 2

        self.nn_hidden_dim = 3

        self.nn_output_dim = 2

        self.actFun_type = 'sigmoid'

        self.nn_layers = 50
        # the nn_layers here are number of hidden_layer plus output_layer

        self.reg_lambda = 0.01