
from __future__ import print_function

import multiprocessing as mp

import keras.objectives
import keras.optimizers
import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from alpha import feature_extractor
from alpha.transform import Transform
from hexmer import get_score_matrix
from predictor import predictor


def _parse_seq(args):
    fe, code, seq = args
    new_dict = fe.extract_features_using_dict(code, seq)
    if 'exception' in new_dict.keys():
        return None
    else:
        return new_dict



class predictor_cv_relu(predictor):


    def __init__(self):
        super.__init__(self)

    def train(self, coding_filename, non_coding_filename, sScoreMatrix=None, save_input=False):
        # 读取数据
        self._logger.info("Reading training_set")
        if sScoreMatrix is None:
            self._logger.info("Calculating the matrix for S-Score")
            sScoreMatrix = get_score_matrix(coding_filename, non_coding_filename)
            self._logger.info("[DONE] Calculating the matrix for S-Score")
        self.fe = fe = feature_extractor(sScoreMatrix)

        with open(coding_filename, "r") as f:
            lines = f.readlines()
        args0 = [(self.fe, lines[i][:-1], lines[i + 1][:-1]) for i in range(0, len(lines), 2)]

        with open(non_coding_filename, "r") as f:
            lines = f.readlines()
        args1 = [(self.fe, lines[i][:-1], lines[i + 1][:-1]) for i in range(0, len(lines), 2)]


        with mp.Pool(processes=8) as pool:
            data0 = pool.map(_parse_seq, args0)
            data0 = [x for x in data0 if not x is None]
            for i in range(len(data0)):
                data0[i]['verdict'] = 0
            data1 = pool.map(_parse_seq, args1)
            data1 = [x for x in data1 if not x is None]
            for i in range(len(data1)):
                data1[i]['verdict'] = 1

        fss = data0 + data1
        df = pd.DataFrame(fss)
        df = df.drop(['ID', 'seq', 'kozak1', 'kozak2'], axis=1)

        if save_input:
            import random
            temp_int = random.randint(1, 65536)
            df.to_csv("data_" + str(temp_int) + ".csv", index_col=False)

        y = np.asarray(df.verdict[:])
        X = np.asarray(df.drop('verdict', axis=1))

        # 归一化
        self.atrans = Transform(X)
        X_train = self.atrans.transform(X)
        y_train = y

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
        encoder_1 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_1))
        decoder_1 = LeakyReLU(alpha=0.3)(Dense(original_dim[0]))
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
        encoder_2 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_2))
        decoder_2 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_1))
        hh = encoder_2(x2)
        h_hat = decoder_2(hh)
        auto_encdoer_2 = Model(x2, h_hat)
        auto_encdoer_2.compile(optimizer="RMSprop", loss=ae_loss)
        auto_encdoer_2.fit(h1, h1, epochs=a_epoach, shuffle=True, verbose=verbose_level)
        encoder2 = Model(x2, hh)
        h2 = encoder2.predict(h1)
        x3 = Input(shape=(hidden_dim_2,))
        encoder_3 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_3))
        decoder_3 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_2))

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
        encoder_1 = Dense(hidden_dim_1)
        encoder_1 = LeakyReLU(alpha=0.3)(encoder_1)
        decoder_1 = LeakyReLU(alpha=0.3)(Dense(original_dim[0]))

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
        encoder_2 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_2))
        decoder_2 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_1))
        hh = encoder_2(x2)
        h_hat = decoder_2(hh)
        auto_encdoer_2 = Model(x2, h_hat)
        auto_encdoer_2.compile(optimizer="RMSprop", loss=ae_loss)
        auto_encdoer_2.fit(h1, h1, epochs=a_epoach, shuffle=True, verbose=verbose_level)
        encoder2 = Model(x2, hh)
        h2 = encoder2.predict(h1)
        x3 = Input(shape=(hidden_dim_2,))
        encoder_3 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_3))
        decoder_3 = LeakyReLU(alpha=0.3)(Dense(hidden_dim_2))

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
