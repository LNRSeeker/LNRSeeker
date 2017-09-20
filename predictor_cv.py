from __future__ import print_function

import keras.objectives
import keras.optimizers
from keras.layers import Dense, Input
from keras.models import Model

from alpha.transform import Transform
from predictor import predictor


class predictor_cv(predictor):
    """
    A predictor specialized for cross-validation.
    """

    def __init__(self):
        super.__init__()

    def train_by_data(self, x_train, y_train):
        """

        :param x_train: the input matrix, which should not be normalized.
        :param y_train: the label for the input
        :return: nothing
        """

        self.atrans = Transform(x_train)
        X_train = self.atrans.transform(x_train)

        # 载入神经网络的最佳参数

        params = self._params

        #    logger.info("Checking the model: \n" + str(params))
        #else:
        #    logger.info("Checking the default model.")

        self._logger.info("Training network")
        verbose_level = 2
        myorg_dim = len(X_train[0])
        original_dim = (len(X_train[0]),)
        latent_dim = original_dim

        if 'hidden_dim_1' in params.keys():
            hidden_dim_1 = params['hidden_dim_1']
        else:
            hidden_dim_1 = 200

        # hidden_dim_1 = {{choice([200, 300, 400, 500])}}

        if 'hidden_dim_2' in params.keys():
            hidden_dim_2 = params['hidden_dim_2']
        else:
            hidden_dim_2 = 200

        if 'hidden_dim_3' in params.keys():
            hidden_dim_3 = params['hidden_dim_3']
        else:
            hidden_dim_3 = 50

        if 'mlp' in params.keys():
            is_mlp = params['mlp']
        else:
            is_mlp = False

        if 'a_epoch' in params.keys():
            a_epoach = params['a_epoch']
        else:
            a_epoach = 15

        if 'mlp_epoch' in params.keys():
            mlp_epoch = params['mlp_epoch']
        else:
            mlp_epoch = 50

        if 'is_relu' in params.keys():
            is_relu = params['is_relu']
        else:
            is_relu = False

        if 'alpha'in params.keys():
            alpha = params['alpha']
        else:
            alpha = 1

        hidden_dim_4 = 10
        nb_epoch = 5

        # 定义网络
        x = Input(shape=original_dim)
        encoder_1 = Dense(hidden_dim_1, activation='sigmoid')
        decoder_1 = Dense(original_dim[0], activation='sigmoid')
        h = encoder_1(x)
        x_hat = decoder_1(h)

        # Autoencoder的损失函数
        def ae_loss(y_true, y_pred):
            original_loss = keras.objectives.mean_squared_error(y_true, y_pred)
            kld_loss = keras.objectives.kld(y_true, y_pred)
            return original_loss + alpha * kld_loss

        auto_encdoer = Model(x, x_hat)
        auto_encdoer.compile(optimizer="RMSprop", loss=ae_loss)
        auto_encdoer.fit(X_train, X_train, epochs=a_epoach, shuffle=True, verbose=verbose_level)

        encoder = Model(x, h)
        h1 = encoder.predict(X_train)
        x2 = Input(shape=(hidden_dim_1,))
        encoder_2 = Dense(hidden_dim_2, activation='sigmoid')
        decoder_2 = Dense(hidden_dim_1, activation='sigmoid')
        hh = encoder_2(x2)
        h_hat = decoder_2(hh)
        auto_encdoer_2 = Model(x2, h_hat)
        auto_encdoer_2.compile(optimizer="RMSprop", loss=ae_loss)
        auto_encdoer_2.fit(h1, h1, epochs=a_epoach, shuffle=True, verbose=verbose_level)
        encoder2 = Model(x2, hh)
        h2 = encoder2.predict(h1)
        x3 = Input(shape=(hidden_dim_2,))
        encoder_3 = Dense(hidden_dim_3, activation='sigmoid')
        decoder_3 = Dense(hidden_dim_2, activation='sigmoid')

        hh3 = encoder_3(x3)
        h_hat = decoder_3(hh3)
        auto_encdoer_3 = Model(x3, h_hat)
        auto_encdoer_3.compile(optimizer="RMSprop", loss=ae_loss)
        auto_encdoer_3.fit(h2, h2, epochs=a_epoach, shuffle=True, verbose=verbose_level)
        hhh = encoder_2(h)
        hhh = encoder_3(hhh)

        if is_relu:
            active_name = 'relu'
        else:
            active_name = 'sigmoid'

        if not is_mlp == 0:
            y = Dense(is_mlp, activation=active_name)(hhh)
            y = Dense(1, activation='sigmoid')(y)
        else:
            y = Dense(1, activation='sigmoid')(hhh)

        self.model = Model(x, y)
        self.model.compile(optimizer="RMSprop", loss=keras.objectives.binary_crossentropy,
                           metrics=["accuracy"])
        self.model.fit(X_train, y_train, epochs=mlp_epoch, shuffle=True, verbose=verbose_level)

        self._logger.info("Training network Done")
