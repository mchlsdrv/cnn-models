# GENERAL
import os
from os import (
    makedirs
)
import time
import gc
from pathlib import Path
import pickle as pkl

# DATA
import numpy as np
import pandas as pd

# PYTORCH
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)
# OPT
import hyperopt
from hyperopt import (
    hp,
    Trials,
    tpe,
    fmin,
    STATUS_OK,
    STATUS_FAIL,
)

from tqdm import trange

# LOCAL
from aux_funcs import (
    get_unbiased_std,
    get_run_time
)
from plot_funcs import (
    plot_binary_metrics,
)


class DataSet(Dataset):
    def __init__(self, params):
        self.X = None
        self.y = None
        self.n_samples = None
        self.dim = None

        self.load_data(**params)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def bin_2_stoch(self, bin_vecs, pos_gate, neg_gate):
        '''
        Converts binary vectors into stochastic, where:
            - 1~N(1, 0.01)
            - 0~N(-1, 0.01)
        '''
        
        # 1) Create stochastic data vectors of zeros
        stoch_vecs = np.random.normal(neg_gate['mu'], neg_gate['sigma'], bin_vecs.shape)

        # 2) Find the highs in the binary data
        highs = np.argwhere(bin_vecs>0)
        high_rows, high_cols = highs[:, 0], highs[:, 1]

        # 3) Change the lows in the stochastic data to highs
        stoch_vecs[high_rows, high_cols] = np.random.normal(pos_gate['mu'], pos_gate['sigma'], highs.shape[0])
        
        return stoch_vecs


    def load_data(self, X, y, stoch_gates=None):    
        assert isinstance(X, np.ndarray) or isinstance(X, torch.Tensor), f"\'X\' must be of type \'np.ndarray\' or of type \'torch.Tensor\', but is of type \'{type(X)}\'!"
        assert isinstance(y, np.ndarray) or isinstance(y, torch.Tensor), f"\'y\' must be of type \'np.ndarray\' or of type \'torch.Tensor\', but is of type \'{type(y)}\'!"
        
        self.X = X
        if stoch_gates is not None:
            self.X = self.bin_2_stoch(bin_vecs=self.X, pos_gate=stoch_gates['pos_gate'], neg_gate=stoch_gates['neg_gate'])

        self.y = y
        if stoch_gates is not None:
            self.y = self.bin_2_stoch(bin_vecs=self.y, pos_gate=stoch_gates['pos_gate'], neg_gate=stoch_gates['neg_gate'])

        self.n_samples = self.X.shape[0]

        self.dim = self.X.shape[1]

    def save_data(self, save_dir, save_name):
        if isinstance(save_dir, Path):
            if not save_dir.is_dir():
                makedirs(save_dir)

            with (save_dir / f'{save_name}_X.npy').open(mode='wb') as npy_out:
                np.save(npy_out, self.X)
            
            with (save_dir / f'{save_name}_y.npy').open(mode='wb') as npy_out:
                np.save(npy_out, self.y)

    def dim(self):
        return self.dim


class IterModelFiter:
    def __init__(self, model):
        self.model = model
        self.results_df = None
        self.search_space = None
    
    def _get_W(self, test=False):
        if test:
            W = np.zeros((self.search_space['n_users'], self.search_space['n_users'])) + self.results_df.loc[:, 'best_W'].values[-1][0]
        else:
            assert self.search_space['W'][0][0] is not None, f'\'W\' vector is None!'
            W = np.zeros((self.search_space['n_users'], self.search_space['n_users'])) + self.search_space['W'][0][0]
        return W

    def loss_min_func(self):
        def _obj_func(params):

            metrics = self.model.eval(
                X=params['X'], 
                y=params['y'],
                a=params['a'], 
                b=params['b'], 
                th=params['th'], 
                P=params['P'],
                P_offset=params['P_offset'], 
                W=params['W'],
                accumulative_activity=params['accumulative_activity'],
                W_offset=params['W_offset'], 
                pos_label=params['pos_label'], 
                average=params['average'], 
                beta=params['beta']
            )

            obj_out = dict(
                loss= -1 * metrics['f_score_avg'],

                status = STATUS_OK,
                
                active_prop_avg = params['y'].mean(),

                precision_avg = metrics['precision_avg'],
                f_score_avg = metrics['f_score_avg'],
                recall_avg = metrics['recall_avg'],

                a = params['a'],
                b = params['b'],
                th = params['th'],

                P = params['P'],
                P_offset = params['P_offset'],

                W = params['W'],
                W_offset = params['W_offset'],

                Z = metrics['Z'], 
            )

            return obj_out

        # 1) Get the Trials() object
        trials = Trials()

        # 2) Minimize the loss
        best_iter = fmin(
            _obj_func, 
            self.search_space, 
            trials=trials, 
            algo=tpe.suggest, 
            max_evals=self.search_space['max_evals'], 
            show_progressbar=False
        )

        # 3) Return the parameters which result in the lowest loss
        return dict(
                loss = trials.best_trial['result']['loss'],
                
                status = 'Pass' if -1 * trials.best_trial['result']['loss'] >= self.search_space['loss_th'] else 'Fail',
                
                active_prop_avg = trials.best_trial['result']['active_prop_avg'],

                best_precision_avg = trials.best_trial['result']['precision_avg'],
                best_f_score_avg = trials.best_trial['result']['f_score_avg'],
                best_recall_avg = trials.best_trial['result']['recall_avg'],
                
                best_a = trials.best_trial['result']['a'],
                best_b = trials.best_trial['result']['b'],
                best_th = trials.best_trial['result']['th'],
                
                best_P = [trials.best_trial['result']['P']],
                best_P_offset = trials.best_trial['result']['P_offset'],
                
                best_W = [trials.best_trial['result']['W']],
                best_W_offset = trials.best_trial['result']['W_offset'],

                Z = [trials.best_trial['result']['Z']], 
            )

    def init_params(self, params):
        columns=[
            'type', 'loss', 'status', 'active_prop_avg',
            'best_precision_avg', 'best_f_score_avg', 'best_recall_avg', 
            'best_a', 'best_b', 'best_th',
            'best_P', 'best_P_offset',
            'best_W', 'best_W_offset',
            'Z'
        ]
        self.results_df = pd.DataFrame(columns=columns)
        
        if params['opt']:
            self.search_space = dict(
                max_evals = params['max_evals'],

                fails = 0,

                std=params['std'],

                n_users=params['data']['n_users'],
                
                accumulative_activity=params['accumulative_activity'],

                a=hp.uniform('a', params['hyperparams']['a'][0], params['hyperparams']['a'][1]) if isinstance(params['hyperparams']['a'], tuple) else params['hyperparams']['a'],
                b=hp.uniform('b', params['hyperparams']['b'][0], params['hyperparams']['b'][1]) if isinstance(params['hyperparams']['b'], tuple) else params['hyperparams']['b'],
                th=hp.uniform('th', params['hyperparams']['th'][0], params['hyperparams']['th'][1]) if isinstance(params['hyperparams']['th'], tuple) else params['hyperparams']['th'],

                loss_th=params['metrics']['loss_th'],
                pos_label=params['metrics']['pos_label'],
                average=params['metrics']['average'],
                beta=params['metrics']['beta'],

                # Priors
                P = [
                    hp.uniform(f'p_{k}', 0., 1.) for k in range(params['data']['n_users'])
                ] if params['data']['P'] is None else params['data']['P'],
                P_offset=hp.uniform('P_offset', params['hyperparams']['P_offset'][0], params['hyperparams']['P_offset'][1]) if isinstance(params['hyperparams']['P_offset'], tuple) else params['hyperparams']['P_offset'],
                
                # Influences
                W = [[
                    hp.uniform(f'w_{k}', 0., 1.) for k in range(params['data']['n_users'])
                ]]*params['data']['n_users'] if params['data']['W'] is None else params['data']['W'],
                W_offset=hp.uniform('W_offset', params['hyperparams']['W_offset'][0], params['hyperparams']['W_offset'][1]) if isinstance(params['hyperparams']['W_offset'], tuple) else params['hyperparams']['W_offset'],
            )
    
    def _update_search_space(self, results):
        self.search_space['fails'] = 0 if results['status'] == 'Pass' else self.search_space['fails'] + 1
        
        self.search_space['a'] = hp.normal('a', results['best_a'], self.search_space['std']) if isinstance(self.search_space['a'], hyperopt.pyll.base.Apply) else self.search_space['a']
        self.search_space['b'] = hp.normal('b', results['best_b'], self.search_space['std']) if isinstance(self.search_space['b'], hyperopt.pyll.base.Apply) else self.search_space['b']
        self.search_space['th'] = hp.uniform('th', np.max([0., results['best_th']-self.search_space['std']]), np.min([1., results['best_th']+self.search_space['std']])) if isinstance(self.search_space['th'], hyperopt.pyll.base.Apply) else self.search_space['th']
                
        # Priors
        self.search_space['P'] = [
            hp.uniform(f'p_{k}', np.max([0., p-self.search_space['std']]), np.min([1., p+self.search_space['std']])) 
            for k, p in enumerate(results['best_P'][0])
        ] if not isinstance(self.search_space['P'], np.ndarray) else self.search_space['P']
        self.search_space['P_offset'] = hp.normal('P_offset', results['best_P_offset'], self.search_space['std']) if isinstance(self.search_space['P_offset'], hyperopt.pyll.base.Apply) else self.search_space['P_offset']

        # Weights
        self.search_space['W'] =  [[
            hp.uniform(f'w_{k}', np.max([0., w-self.search_space['std']]), np.min([1., w+self.search_space['std']])) 
            for k, w in enumerate(results['best_W'][0][0])
        ]]*self.search_space['n_users'] if not isinstance(self.search_space['W'], np.ndarray) else self.search_space['W']
        self.search_space['W_offset'] = hp.normal('W_offset', results['best_W_offset'], self.search_space['std']) if isinstance(self.search_space['W_offset'], hyperopt.pyll.base.Apply) else self.search_space['W_offset']

    def fit(self, params):
        start_time = time.time()
        params['opt'] = 'train_data' in params['data'].keys()
        self.init_params(params=params)
        # 1) Build Data Sets
        # Train
        train_means = None
        train_stds = None
        if 'train_data' in params['data'].keys():
            train_dataset = DataSet(
                params=dict(
                    X=params['data']['train_data']['X'],
                    y=params['data']['train_data']['y'],
                )
            )
            train_data_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=params['data_loader']['train_btch_sz'],
                shuffle=params['data_loader']['shuffle'],
                num_workers=params['data_loader']['num_workers'],
                drop_last=params['data_loader']['drop_last']
            )

            train_precision_means = np.array([])
            train_precision_stds = np.array([])

            train_f_score_means = np.array([])
            train_f_score_stds = np.array([])

            train_recall_means = np.array([])
            train_recall_stds = np.array([])

            epoch_progress_bar = trange(params['epochs'], position=0, leave=True)
            epoch_train_times = []
            for epoch in epoch_progress_bar:
                epoch_start_time = time.time()

                batch_precision_avgs = np.array([])
                batch_f_score_avgs = np.array([])
                batch_recall_avgs = np.array([])

                for btch_idx, (X, y) in enumerate(train_data_loader):

                    self.search_space['X'], self.search_space['y'] = X.numpy(), y.numpy()
                    results = self.loss_min_func()
                    self._update_search_space(results=results)

                    batch_precision_avgs = np.append(batch_precision_avgs, results['best_precision_avg'])
                    batch_f_score_avgs = np.append(batch_f_score_avgs, results['best_f_score_avg'])
                    batch_recall_avgs = np.append(batch_recall_avgs, results['best_recall_avg'])
                    
                    results['type'] = 'Train'
                    self.results_df = self.results_df.append(pd.DataFrame(results))

                train_precision_means = np.append(train_precision_means, batch_precision_avgs.mean())
                train_precision_stds = np.append(train_precision_stds, batch_precision_avgs.std())

                train_f_score_means = np.append(train_f_score_means, batch_f_score_avgs.mean())
                train_f_score_stds = np.append(train_f_score_stds, batch_f_score_avgs.std())

                train_recall_means = np.append(train_recall_means, batch_recall_avgs.mean())
                train_recall_stds = np.append(train_recall_stds, batch_recall_avgs.std())

                epoch_progress_bar.set_description(f"| Precision: {train_precision_means[-1]:.2f}+/-{train_precision_stds[-1]:.4f} | F Score: {train_f_score_means[-1]:.2f}+/-{train_f_score_stds[-1]:.4f} | Recall: {train_recall_means[-1]:.2f}+/-{train_recall_stds[-1]:.4f} |")
                
                epoch_train_times.append(time.time() - epoch_start_time)

            train_means = [train_precision_means.mean(), train_f_score_means.mean(), train_recall_means.mean()]
            train_stds = [get_unbiased_std(std_arr=train_precision_stds), get_unbiased_std(std_arr=train_f_score_stds), get_unbiased_std(std_arr=train_recall_stds)]
            
        # Test
        test_means = None
        test_stds = None
        if 'test_data' in params['data'].keys():
            test_dataset = DataSet(
                params=dict(
                    X=params['data']['test_data']['X'],
                    y=params['data']['test_data']['y'],
                )
            )
            test_data_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=params['data_loader']['test_btch_sz'],
                shuffle=params['data_loader']['shuffle'],
                num_workers=params['data_loader']['num_workers'],
                drop_last=params['data_loader']['drop_last']
            )
            test_precision_avgs = np.array([])
            test_f_score_avgs = np.array([])
            test_recall_avgs = np.array([])

            epoch_test_times = []
            for btch_idx, (X, y) in enumerate(test_data_loader):
                
                epoch_start_time = time.time()

                test_results = self.model.eval(
                    
                    X=X.numpy(), 
                    y=y.numpy(),
                    
                    accumulative_activity=params['accumulative_activity'],

                    p=params['p'] if 'p' in params.keys() else None, 

                    a=self.results_df.loc[:, 'best_a'].values[-1] if self.results_df.shape[0] > 0 else None, 
                    b=self.results_df.loc[:, 'best_b'].values[-1] if self.results_df.shape[0] > 0 else None, 
                    th=self.results_df.loc[:, 'best_th'].values[-1] if self.results_df.shape[0] > 0 else None, 
                    
                    P=self.results_df.loc[:, 'best_P'].values[-1] if self.results_df.shape[0] > 0 else None,
                    P_offset=self.results_df.loc[:, 'best_P_offset'].values[-1] if self.results_df.shape[0] > 0 else None, 
                    
                    W=self.results_df.loc[:, 'best_W'].values[-1] if self.results_df.shape[0] > 0 else None,
                    W_offset=self.results_df.loc[:, 'best_W_offset'].values[-1] if self.results_df.shape[0] > 0 else None, 
                    
                    pos_label=params['metrics']['pos_label'], 
                    average=params['metrics']['average'], 
                    beta=params['metrics']['beta']
                )

                test_precision_avgs = np.append(test_precision_avgs, test_results['precision_avg'])
                test_f_score_avgs = np.append(test_f_score_avgs, test_results['f_score_avg'])
                test_recall_avgs = np.append(test_recall_avgs, test_results['recall_avg'])

                results = dict(
                    type='Test',

                    loss = None,
                    
                    status = 'Pass',
                    
                    active_prop_avg = y.numpy().mean(),
                    
                    best_precision_avg = test_results['precision_avg'],
                    best_f_score_avg = test_results['f_score_avg'],
                    best_recall_avg = test_results['recall_avg'],

                    best_a = self.results_df.loc[:, 'best_a'].values[-1] if self.results_df.shape[0] > 0 else None ,
                    best_b = self.results_df.loc[:, 'best_b'].values[-1] if self.results_df.shape[0] > 0 else None ,
                    best_th = self.results_df.loc[:, 'best_th'].values[-1] if self.results_df.shape[0] > 0 else None ,
                    
                    best_P = [self.results_df.loc[:, 'best_P'].values[-1]] if self.results_df.shape[0] > 0 else [None],
                    best_P_offset=self.results_df.loc[:, 'best_P_offset'].values[-1] if self.results_df.shape[0] > 0 else None ,
                    
                    best_W = [self.results_df.loc[:, 'best_W'].values[-1]] if self.results_df.shape[0] > 0 else [None],
                    best_W_offset=self.results_df.loc[:, 'best_W_offset'].values[-1] if self.results_df.shape[0] > 0 else None ,

                    Z = [test_results['Z']], 
                )

                self.results_df = self.results_df.append(pd.DataFrame(results))
                
                epoch_test_times.append(time.time() - epoch_start_time)

            test_means = [test_precision_avgs.mean(), test_f_score_avgs.mean(), test_recall_avgs.mean()]
            test_stds = [test_precision_avgs.std(), test_f_score_avgs.std(), test_recall_avgs.std()]

        print('Final Stats:')
        if train_means is not None and train_stds is not None:
            epoch_train_times = np.array(epoch_train_times)
            print(f'''
            > Train:
                - Precision: {train_means[0]:.4f}+/-{train_stds[0]:.4f}
                - F Score: {train_means[1]:.4f}+/-{train_stds[1]:.4f}
                - Recall: {train_means[2]:.4f}+/-{train_stds[2]:.4f}
                - Runtime: {get_run_time(epoch_train_times.mean())}+/-{get_run_time(epoch_train_times.std())} for iteration
            ''')
        if test_means is not None and test_stds is not None:
            epoch_test_times = np.array(epoch_test_times)
            print(f'''
            > Test:
                - Precision: {test_means[0]:.4f}+/-{test_stds[0]:.4f}
                - F Score: {test_means[1]:.4f}+/-{test_stds[1]:.4f}
                - Recall: {test_means[2]:.4f}+/-{test_stds[2]:.4f}
                - Runtime: {get_run_time(epoch_test_times.mean())}+/-{get_run_time(epoch_test_times.std())} for iteration
            ''')
        self.results_df.to_pickle(params['results_dir']/f"{self.model.description} results df.pkl")
        plot_binary_metrics(
            metric_types=['Precision', 'F-Score', 'Recall'], 
            train=train_means, train_err=train_stds,
            val=None, val_err=None,
            test=test_means, test_err=test_stds, 
            x_lab='Metric Type', y_lab='Precision / F-Score / Recall', 
            title=f"{self.model.description} Model Metrics", save_dir=params['results_dir']/'[PLOTS]', save_name=f"{self.model.description} model metrics plot"
        )

        print(f'> Total Runtime: {get_run_time(time.time() - start_time)}')

        return test_means
