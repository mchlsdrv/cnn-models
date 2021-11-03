import numpy as np
import matplotlib.pyplot as plt
from pathlib import (
    Path
)

# LOCAL
from aux_funcs import (
    check_dir,
    get_ts,
)


def show_bar_val(bar_data, ax):
    for bar in bar_data:
        height = bar.get_height()
        ax.annotate(
            f'{height:.4f}',
            xy=(bar.get_x()+bar.get_width(), height),
            xytext=(-1*bar.get_width(), 3), # 3 points vertical offset
            textcoords='offset points',
            ha='right',
            va='bottom'
        )


def plot_binary_metrics(metric_types, train, train_err=None, val=None, val_err=None, test=None, test_err=None, x_lab='Metric', y_lab='Precision / F-Score / Recall', title='Binary Metrics Plot', save_dir=None, save_name='binary_metrics_plot'):
    metric_loc = np.arange(len(metric_types))
    # 1) Configure bar x locations
    if (train is not None) and (val is not None) and (test is not None):
        width = .25
        train_x = metric_loc-width
        val_x = metric_loc
        test_x = metric_loc+width
    elif (train is not None and val is not None) or \
         (train is not None and test is not None) or\
         (val is not None and test is not None):
        width = .35
        train_x = metric_loc-width/2
        val_x = metric_loc+width/2
        test_x = metric_loc+width/2
    else:
        width = .5
        train_x = metric_loc
        val_x = metric_loc
        test_x = metric_loc
    
    # 2) Add bars to list
    bars = []
    fig, ax = plt.subplots()
    if train is not None:
        bars.append(
            ax.bar(
                x=train_x, 
                height=train,
                yerr=train_err,
                width=width,
                label='Train'
            )
        )
    if val is not None:
        bars.append(
            ax.bar(
                x=val_x, 
                height=val,
                yerr=val_err,
                width=width,
                label='Val'
            )
        )
    if test is not None:
        bars.append(
            ax.bar(
                x=test_x, 
                height=test,
                yerr=test_err,
                width=width,
                label='Test'
            )
        )

    # 3) Configure plot layout
    ax.set_title(title)
    
    # 3.1) x
    ax.set_xlabel(x_lab)
    ax.set_xticks(metric_loc)
    ax.set_xticklabels(metric_types)
    
    # 3.2) y
    ax.set_ylabel(y_lab)
    ax.set_ylim([0., 1.])

    ax.grid(True)
    ax.legend()
    
    # 4) Add bar values
    for bar in bars:
        show_bar_val(bar_data=bar, ax=ax)

    fig.tight_layout()
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # 5) Add bar values
    if check_dir(dir_path=save_dir):
        plt.savefig(save_dir / f'{save_name}.png')

    plt.show()


def plot_counts(plot_params_list, title):
    
    if len(plot_params_list) > 1:
        fig, ax = plt.subplots(len(plot_params_list), 1, figsize=(15*len(plot_params_list), 18))
        for ax_idx, plot_params in enumerate(plot_params_list):
            plot_data = ax[ax_idx].bar(
                x=plot_params['x_vals'], 
                height=plot_params['values'],
                width=0.15,
                label=plot_params['label']
            )
            show_bar_val(bar_data=plot_data, ax=ax[ax_idx])
            ax[ax_idx].set(title=plot_params['title'], xlabel=plot_params['xlabel'], ylabel=plot_params['ylabel'])
    elif len(plot_params_list) == 1:
        fig, ax = plt.subplots(len(plot_params_list), 1, figsize=(20, 10))
        plot_data = ax.bar(
            x=plot_params_list[0]['x_vals'], 
            height=plot_params_list[0]['values'],
            width=0.15,
            label=plot_params_list[0]['label']
        )
        show_bar_val(bar_data=plot_data, ax=ax)
        ax.set(title=plot_params_list[0]['title'], xlabel=plot_params_list[0]['xlabel'], ylabel=plot_params_list[0]['ylabel'])
    
    if len(plot_params_list) > 0:
        fig.suptitle(title, fontsize='large')

        plt.show()


def plot_fit_history(train, val, train_err=None, val_err=None, train_lab='Train', val_lab='Validation', x_lab='Epoch', y_lab='Loss', title='Train / Validation Loss Plot', save_dir=None, save_name='train_val_loss_plot'):
    plt.figure(figsize=(20, 10))
    plt.errorbar(x=np.arange(train.shape[0]), y=train, 
        yerr=train_err, label=train_lab)
    plt.errorbar(x=np.arange(val.shape[0]), y=val, 
        yerr=val_err, label=val_lab)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    plt.grid(True)

    if check_dir(dir_path=save_dir):
        plt.savefig(save_dir / f'{save_name}.png')

    plt.show()


def plot_grad_flow(named_params, title, save_dir, save_name):
    def _format_name(layer_name):
        clean_name = layer_name[::-1][layer_name[::-1].index('.')+1:][::-1]
        net_num = clean_name[:clean_name.index('.')]
        layer_num = clean_name[::-1][:clean_name[::-1].index('.')][::-1]
        return f'net{net_num}_l{layer_num}'

    avg_grads = []
    layers = []
    for name, param in named_params:
        if param.requires_grad and 'bias' not in name:
            weight_name = _format_name(name)
            layers.append(weight_name)
            # layers.append(name)
            avg_grads.append(param.grad.abs().mean())
    n_grads = len(avg_grads)
    plt.plot(avg_grads, alpha=.3, color='b')
    plt.hlines(0, 0, n_grads+1, linewidth=1, color='k')
    plt.xticks(range(0, n_grads), layers, rotation='vertical')
    plt.xlim(xmin=0, xmax=n_grads)
    plt.xlabel('layers')
    plt.ylabel('average gradient')
    plt.title(title)
    plt.grid(True)
    if isinstance(save_dir, Path):
        plt.savefig(save_dir / f'{save_name}_grad_flow_{get_ts()}.png')
    else:
        plt.show()
