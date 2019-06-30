from Loader import Loader
from model import unet


if __name__ == '__main__':
    loader = Loader(batch_size=2)
    Net = unet()
    for image_batch, seg in loader.get_minibatch():
       # results = Net.predict(image_batch, 2, verbose=1)
        results2 = Net(image_batch)


