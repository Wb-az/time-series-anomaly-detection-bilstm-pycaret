import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ['view_per_channel', 'view_all', 'plot_predicted_anomalies',
           'plot_anomaly_threshold', 'plot_loss', 'plot_raw_data',
           'plot_reconstruction', 'plot_scores_distribution',
           'bilstm_predicted_anomalies', 'save_plot', 'scatter_view',
           'scatter_anomalies_plot']

plots_dir = os.path.join(os.getcwd(), 'plots')

os.makedirs(plots_dir, exist_ok=True)


def view_per_channel(df, plot_name=None, outdir=plots_dir):
    """
    :param outdir: str path to save the plot
    :param plot_name: str name of the plot
    :param df: dataframe of the data
    :return: show plot
    """

    figure, axes = plt.subplots(len(df.columns), 1, sharey=True, figsize=(8, 10))
    colors = ['r', 'b', 'g', 'm', 'blue', 'orange', 'olive', 'purple']
    plt.subplots_adjust(hspace=0.5)
    for i, c in enumerate(df.columns):
        axes[i].plot(df[c], color=colors[i])
        axes[i].grid(True, ls='--', lw=0.5)
        axes[i].set_ylabel(c)

    figure.suptitle(' Bearings Vibrations')
    plt.xlabel('date')
    save_plot(outdir=outdir, plot_name=plot_name)
    return plt.show(block=False)


def view_all(df, outdir=plots_dir, plot_name=None, size=(8, 2.5)):
    """
    :param df: dataframe sataset
    :param outdir: path to save the plot
    :param plot_name: str plot name
    :param size: tuple for the figure size
    :return: show the plot
    """
    df.plot(figsize=size, xlabel='date', ylabel='average signal', grid=True)
    plt.grid(ls='--', lw=0.5)
    save_plot(outdir, plot_name)
    return plt.show()


def save_plot(outdir=plots_dir, plot_name=None):
    """
    :param outdir: a string for the directory to save the figure
    :param plot_name: a string with the name to save the figure
    :return: save the plot
    """
    return plt.savefig(os.path.join(outdir, f'{plot_name}.png'), bbox_inches='tight',
                       format='png', dpi=800)


def plot_raw_data(df=None, sample_rate=20480, suptitle=None, xlabel='timestamp'):
    """
    :param df: dataframe dataset
    :param sample_rate: int sampling rate (Hz)-frequency
    :param subtitle: str a string with the plot title
    :param xlabel: str axis s label
    :return: display plot
    """

    fig, axes = plt.subplots(df.shape[1], 1, sharex=True, sharey=True, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.5)
    for i, c in enumerate(df.columns):
        rolling_avg = df[c].rolling(sample_rate).mean()
        axes[i].plot(np.arange(0, len(df), dtype='float64') / sample_rate, rolling_avg, alpha=0.5)
        axes[i].grid(True)
        axes[i].set_ylabel(f"{c}")
    fig.suptitle(suptitle)
    plt.xlabel(xlabel)

    return plt.show(block=False)


def scatter_view(df, plot_name=None, outdir=plots_dir, model_name=None):
    """
    :param df: dataframe of the data
    :param outdir: str path to save the plot
    :param plot_name: str name to save the plot
    :param model_name: str model name
    :return: show plot
    """

    fig, ax = plt.subplots(figsize=(8, 2.5))
    colors = ['r', 'b', 'g', 'm', 'blue', 'orange', 'olive', 'purple']

    for i, c in enumerate(df.columns):
        ax.scatter(df.index, df[c], s=20, c=colors[i], label=c)

    plt.grid(ls='--', lw=0.3)
    plt.xlabel('timestamp')
    plt.legend()
    plt.title(model_name)
    save_plot(outdir=outdir, plot_name=plot_name)

    return plt.show(block=False)


def scatter_anomalies_plot(df=None, plot_name=None, outdir=plots_dir):
    """"
    :param df: dataframe of the data
    :param outdir: str path to save the plot
    :param plot_name: str name of the plot
    :param outdir: str path to save the plot
    :return: show plot
    """

    fig, axes = plt.subplots(len(df.columns), 1, sharey=True, sharex=True, figsize=(8, 15))
    colors = ['r', 'b', 'g', 'm', 'blue', 'orange', 'olive', 'purple']
    plt.subplots_adjust(hspace=0.5)

    for i, c in enumerate(df.columns):
        color = np.where(df[c] == 1, 'k', colors[i])

        axes[i].scatter(df.index, df[c], s=50, label=c, facecolors='none',
                        edgecolors=color, marker='o')
        axes[i].grid(True, ls='--', lw=0.5, c='k')
        axes[i].set(yticks=[0, 1], yticklabels=['normal', 'anomaly'])
        axes[i].legend()

    fig.suptitle('Anomaly detection')
    plt.xlabel('timestamp')

    save_plot(outdir=outdir, plot_name=f'scatter_{plot_name}')
    return plt.show(block=False)


def plot_scores_distribution(df=None, size=(4, 8), outdir=plots_dir, plot_name=None):
    """
    :param df: dataframe of the data to plot
    :param size: a tuple with the figure size
    :param outdir: str path to save the plot
    :param plot_name: str name of the plot
    :return: a plot
    """
    custom_palette = ['#0E38C8', '#FFC300', '#C70039', '#0AB412', '#1DF6E6', '#B40A64']
    sns.set(rc={'figure.figsize': size})
    sns.set(style='whitegrid', font_scale=0.9)
    sns.set_palette(sns.color_palette(custom_palette))

    df_scores = df.melt(var_name='model', value_name='score')
    sns.displot(data=df_scores, kind='kde', col='model', col_wrap=3, x='score', height=3,
                hue='model', facet_kws={'sharey': False, 'sharex': False})

    save_plot(outdir=outdir, plot_name=f'distribution_{plot_name}')

    return plt.show()


def plot_loss(stats_log, model, size=(6, 3), figs_dir=plots_dir):
    """
    :param stats_log: named tuple with the stats
    :param model: a string with the name of the trained model
    :param size: a tuple with the figure size
    :param figs_dir a string to save the metrics plots
    :return: plots of the training metrics
    """

    fig = plt.figure(figsize=size)
    plt.plot(stats_log.train_loss, label='train', color='blue')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(color='k', ls='--', lw=0.5)

    fname = f'{model}_loss.png'
    fig.savefig(os.path.join(figs_dir, fname), bbox_inches='tight', format='png', dpi=700)

    return fig.show()


def plot_predicted_anomalies(df1, df2, size=(12, 7), plot_name=None, outdir=plots_dir):
    """
    :param df1: results dataframe
    :param df2: dataframe to get the indexes from
    :param size: a tuple with the figure size
    :param plot_name: str name of the plot
    :param outdir: str path to save the plot
    :return: show a plot
    """
    figure, axes = plt.subplots(len(df1.columns), 1, sharey=False, figsize=size)

    colors = ['blue', 'green', 'orange', 'olive', 'purple']
    plt.subplots_adjust(hspace=0.5)

    for i, c in enumerate(df1.columns):
        axes[i].plot(df1[c], color=colors[i], label=c)
        indexes = df2.iloc[:, i][df2.iloc[:, i] == 1].index

        axes[i].scatter(indexes, df1[c].loc[indexes], s=20, color='red',
                        edgecolors='red', marker='o', facecolors='none')
        axes[i].grid(True, ls='--', lw=0.5, c='k')
        axes[i].legend()

    save_plot(outdir=outdir, plot_name=f'merged_{plot_name}')


def plot_anomaly_threshold(results_df, thresh=None, size=(8, 2.5), plot_name=None,
                           outdir=plots_dir):
    """
    :param results_df: dataframe - loss metrics
    :param plot_name: str name to save the plot
    :param outdir: str path to save the plot
    :param size: tuple size of the plot
    :param thresh: float - threshold from the predictions distributions
    :return: show a plot
    """

    fig = plt.figure(figsize=size)
    plt.plot(results_df.index, results_df.iloc[:, 0], label=results_df.columns[0], color='green')
    plt.plot(results_df.index, thresh, label=f'thresh {thresh[0]}', color='red')

    plt.grid(color='k', ls='--', lw=0.5)
    plt.xlabel('timestamp')
    plt.xticks(rotation=45)
    plt.legend()
    save_plot(outdir=outdir, plot_name=plot_name)

    return fig.show()


def bilstm_predicted_anomalies(df1, size=(8, 2.5), plot_name=None, outdir=plots_dir):

    fig = plt.figure(figsize=size)
    plt.subplots_adjust(hspace=0.5)

    plt.plot(df1.index, df1.iloc[:, 0], color='green', label=df1.columns[0])
    indexes = df1[df1.iloc[:, 1] == 1].index
    plt.scatter(indexes, df1.loc[indexes][df1.columns[0]], label='anomaly', s=20, color='red',
                edgecolors='red', marker='o', facecolors='none')
    plt.grid(True, ls='--', lw=0.5, c='k')
    plt.legend()

    save_plot(outdir=outdir, plot_name=f'merged_{plot_name}')

    return fig.show()


def plot_reconstruction(data=None, preds=None, plot_name=None,
                        outdir=plots_dir):
    fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, sharey=True,
                             sharex=True, figsize=(8, 10))

    colors = ['blue', 'green', 'orange', 'olive', 'purple']

    plt.subplots_adjust(hspace=0.5)

    for i, col in enumerate(data.columns):
        axes[i].plot(data.index, data[col], label='true', color=colors[i])
        axes[i].plot(data.index, preds[:, i], label='reconstructed', color='cyan', alpha=0.5)
        axes[i].set_title(f'{col}')
        axes[i].legend()
        axes[i].grid(lw=0.5, ls='--', c='k')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    save_plot(outdir=outdir, plot_name=f'scatter_{plot_name}')
