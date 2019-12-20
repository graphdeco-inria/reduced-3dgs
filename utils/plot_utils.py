import matplotlib.pyplot as plt


def get_fig_2d_array_of_images(fig, array_of_images):
    fig.tight_layout()

    rows = len(array_of_images)
    columns = len(array_of_images[0])

    for j in range(rows):
        for i in range(columns):
            fig.add_subplot(rows, columns, j * columns + i + 1)
            plt.imshow(array_of_images[j][i])
    return fig


def plot_loss_lines(lines):
    for line in lines:
        # Converts a list of (x,y) tuples to two lists of x and y.
        x_axis, y_axis = map(list, zip(*line))
        plt.plot(x_axis, y_axis)
