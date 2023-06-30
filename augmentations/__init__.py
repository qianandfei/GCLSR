from .simsiam_aug import SimSiamTransform
def get_aug(name, image_size, train, train_classifier=True):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)

        else:
            raise NotImplementedError
    else:
        raise Exception
    
    return augmentation








