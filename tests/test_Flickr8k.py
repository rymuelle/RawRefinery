from RawRefinery.train.pretraining import Flickr8kDataset

def test_pretraing_ds():
    dataset = Flickr8kDataset()

    img = dataset[0][2]
    assert img.min() >= 0., "Image has range under 0"
    assert img.max() <= 1.0, "Image has range over 1"