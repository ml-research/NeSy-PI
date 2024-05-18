# Create by J.Sha on 02.02.2023
import os
import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import config

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def plot_line_chart(data, path, labels, x=None, title=None, x_scale=None, y_scale=None, y_label=None, show=False,
                    log_y=False, cla_leg=False):
    if data.shape[1] <= 1:
        return

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for i, row in enumerate(data):
        if x is None:
            x = np.arange(row.shape[0]) * x_scale[1]
        y = row
        plt.plot(x, y, label=labels[i], lw=3)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}_{date_now}_{time_now}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()

    if show:
        plt.show()


def plot_scatter_chart(data_list, path, title=None, x_scale=None, y_scale=None,
                       sub_folder=None, labels=None,
                       x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    no_of_colors = len(data_list)
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(no_of_colors)]

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    # for i, data in enumerate(data_list):
    data_x = data_list[:, 0]
    data_y = data_list[:, 1]
    sc = ax1.scatter(data_x, data_y, label="PN Point")
    # plt.colorbar(sc)

    if labels is not None:
        fig.text(0.1, 0.05, labels,
                 bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)), fontsize=15)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

    if log_y:
        plt.yscale('log')
    if log_x:
        plt.xscale('log')

    plt.legend()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.figure(figsize=(1000, 1000 * 0.618))
    if not os.path.exists(str(path)):
        os.mkdir(path)

    img_folder = path / f"{date_now}_{time_now}"
    output_folder = img_folder
    if not os.path.exists(str(img_folder)):
        os.mkdir(str(img_folder))

    if sub_folder is not None:
        if not os.path.exists(str(img_folder / sub_folder)):
            output_folder = img_folder / sub_folder
            os.mkdir(str(output_folder))
        else:
            output_folder = img_folder / sub_folder

    plt.savefig(str(output_folder / f"{title}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def plot_scatter_heat_chart(data_list, path, title=None, x_scale=None, y_scale=None,
                            sub_folder=None, labels=None,
                            x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=False):
    resolution = 2

    def heatmap2d(arr: np.ndarray):
        img = ax1.imshow(arr, cmap='viridis')
        plt.colorbar(img)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    if title is not None:
        ax1.set_title(title)
    if y_label is not None:
        ax1.set_ylabel(y_label)
    if x_label is not None:
        ax1.set_xlabel(x_label)
    if log_y:
        ax1.set_yscale('log')
    if log_x:
        ax1.set_xscale('log')

    # for i, data in enumerate(data_list):
    #     data_map = np.zeros(shape=[resolution, resolution])
    #     for index in range(len(data)):
    #         x_index = int(data[index] * resolution)
    #         y_index = int(data[index] * resolution)
    #         data_map[x_index, y_index] += 1
    data_map = np.zeros(shape=[resolution, resolution])
    data_map[0, 0] = float(data_list[0])
    data_map[0, 1] = float(data_list[1])
    data_map[1, 0] = float(data_list[2])
    data_map[1, 1] = float(data_list[3])
    heatmap2d(data_map)

    if labels is not None:
        fig.text(0.1, 0.05, labels,
                 bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)), fontsize=15)

    # plt.legend()

    # plt.figure(figsize=(1000, 1000 * 0.618))

    if not os.path.exists(str(path)):
        os.mkdir(path)

    img_folder = path / f"{date_now}_{time_now}"
    output_folder = img_folder
    if not os.path.exists(str(img_folder)):
        os.mkdir(str(img_folder))

    if sub_folder is not None:
        if not os.path.exists(str(img_folder / sub_folder)):
            output_folder = img_folder / sub_folder
            os.mkdir(str(output_folder))
        else:
            output_folder = img_folder / sub_folder

    plt.savefig(str(output_folder / f"{title}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def plot_4_zone(is_plot_4zone, B_new, four_scores, all_scores, step):
    if is_plot_4zone:
        for i, clause in enumerate(B_new):
            plot_scatter_heat_chart(four_scores[i],
                                    config.buffer_path / "img",
                                    f"heat_ce_all_{len(B_new)}_{i}",
                                    sub_folder=str(step),
                                    labels=f"{str(clause)}",
                                    x_label="positive score", y_label="negative score")

            plot_scatter_chart(all_scores[i], config.buffer_path / "img", title=f"scatter_all_{len(B_new)}_{i}",
                               x_scale=None, y_scale=None,
                               sub_folder=str(step), labels=f"{str(clause)}",
                               x_label=None, y_label=None, show=False, log_y=False, log_x=False, cla_leg=True)


exp_result_data = {
    "PI_Train": [12.84, 22.75, 0, 2.90, 4.02, 12.69],
    "NoPI_Train": [24.95, 21.30, 0, 24.72, 67.66, 37.04],
    "PI_Group": [8.24, 20.39, 0, 21.71, 10.99, 9.55],
    "NoPI_Group": [12.92, 20.68, 0, 42.14, 21.43, 21.52]
}

plot_line_chart(np.array(list(exp_result_data.values())), config.buffer_path, labels=list(exp_result_data.keys()))
