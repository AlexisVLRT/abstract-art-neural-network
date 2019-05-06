import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import noise
import png
import time


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


class NN(nn.Module):

    def __init__(self, activation=nn.Tanh, num_neurons=16, num_layers=9):
        """
        num_layers must be at least two
        """
        super(NN, self).__init__()
        layers = [nn.Linear(2, num_neurons, bias=True), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons, bias=False), activation()]
        layers += [nn.Linear(num_neurons, 3, bias=False), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def gen_new_image(size_x, size_y, save=True, **kwargs):
    net = NN(**kwargs)
    net.apply(init_normal)
    colors = run_net(net, size_x, size_y)
    plot_colors(colors)
    if save is True:
        save_colors(colors)
    return net, colors


def run_net(net, size_x=128, size_y=128):
    x = np.arange(0, size_x, 1)
    y = np.arange(0, size_y, 1)
    colors = np.zeros((size_x, size_y, 2))
    for i in x:
        for j in y:
            colors[i][j] = np.array([float(i) / size_y - 0.5, float(j) / size_x - 0.5])
    colors = colors.reshape(size_x * size_y, 2)
    img = net(torch.tensor(colors).type(torch.FloatTensor)).detach().numpy()
    return img.reshape(size_x, size_y, 3)


def plot_colors(colors, fig_size=4):
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(colors, interpolation='nearest', vmin=0, vmax=1)


def save_colors(colors):
    plt.imsave(str(np.random.randint(100000)) + ".png", colors)


def run_plot_save(net, size_x, size_y, fig_size=8):
    colors = run_net(net, size_x, size_y)
    plot_colors(colors, fig_size)
    save_colors(colors)


resolution = 720
aspect_ratio = 16 / 9
n_frames = 30 * 34
fps = 30
num_neurons = 32
num_layers = 8
jitter_factor = 0.5  # The lower the more jittery
noise_scale = 5  # The higher the more movement
plot = True

net = NN(num_neurons=num_neurons)
net.apply(init_normal)
perlin = [[[noise.pnoise3(i / (jitter_factor * n_frames), j / num_neurons, k / num_layers, base=j, repeatx=int(1 / jitter_factor), octaves=2) * noise_scale for i in range(n_frames)] for j in range(num_neurons)] for k in range(num_layers)]

for i in range(n_frames):
    start = time.time()
    print('frame {}/{}'.format(i + 1, n_frames))
    for j in range(len(perlin)):
        for k in range(2, num_layers):
            net.state_dict()['layers.{}.weight'.format((k + 1) * 2)][0][j] = perlin[k][j][i]

    colors = run_net(net, resolution, int(aspect_ratio * resolution))

    colors[0][0] = 1
    colors[-1][-1] = 0
    colors = (colors * 255).round().astype(np.uint8)
    png.fromarray(colors, 'RGB').save('frames/out{}.png'.format(i))

    if plot:
        plt.clf()
        plt.imshow(colors)
        plt.pause(0.0001)

    print('Done in {}s'.format(round(time.time() - start, 1)))
