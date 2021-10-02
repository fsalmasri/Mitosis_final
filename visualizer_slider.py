import numpy as np
import matplotlib.pyplot as plt

class visualizer_slider:
    def __init__(self, im_bmp, crop_size):
        self.im_bmp = im_bmp
        self.crop_size = crop_size
        self.h, self.w, _ = self.im_bmp.shape
        self.h_blocks = int(np.ceil(self.h / self.crop_size))
        self.w_blocks = int(np.ceil(self.w / self.crop_size))


        self.arr = []
        self.recvored_im = np.zeros_like(im_bmp)
        self.recvored_mak = np.zeros((self.h, self.w, 1))



    def make_tiles(self):
        for i in range(self.h_blocks):
            x_st = i * self.crop_size
            x_end = x_st + self.crop_size

            if x_end > self.h:
                x_st = self.h - self.crop_size
                x_end = self.h

            for j in range(self.w_blocks):
                y_st = j * self.crop_size
                y_end = y_st + self.crop_size

                if y_end > self.h:
                    im = self.im_bmp[x_st: x_end, -self.crop_size:]
                else:
                    im = self.im_bmp[x_st: x_end, y_st: y_end]

                self.arr.append(im)

    def recover_image(self):
        c = 0
        for i in range(self.h_blocks):
            x_st = i * self.crop_size
            x_end = x_st + self.crop_size

            if x_end > self.h:
                x_st = self.h - self.crop_size
                x_end = self.h

            for j in range(self.w_blocks):
                y_st = j * self.crop_size
                y_end = y_st + self.crop_size

                if y_end > self.h:
                    self.recvored_im[x_st: x_end, -self.crop_size:] = self.arr[c]
                else:
                    self.recvored_im[x_st: x_end, y_st: y_end] = self.arr[c]
                c += 1

    def recover_mask(self):
        c = 0
        for i in range(self.h_blocks):
            x_st = i * self.crop_size
            x_end = x_st + self.crop_size

            if x_end > self.h:
                x_st = self.h - self.crop_size
                x_end = self.h

            for j in range(self.w_blocks):
                y_st = j * self.crop_size
                y_end = y_st + self.crop_size

                if y_end > self.h:
                    self.recvored_mak[x_st: x_end, -self.crop_size:] = self.arr[c]
                else:
                    self.recvored_mak[x_st: x_end, y_st: y_end] = self.arr[c]
                c += 1

    def plot_tiles(self):

        fig, axs = plt.subplots(5, 5, figsize=(15, 15), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.1, wspace=.1)

        axs = axs.ravel()

        for i in range(len(self.arr)):
            axs[i].imshow(self.arr[i])
            axs[i].set_title(str(i))

        plt.show()

    def plot_image(self):
        plt.figure(figsize=(15, 10))
        plt.imshow(self.im_bmp)
        plt.show()

    def plot_recovered(self):
        plt.figure(figsize=(15, 10))
        plt.imshow(self.recvored_im)
        plt.show()
