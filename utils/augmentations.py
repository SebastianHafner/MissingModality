import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


def compose_transformations(cfg, no_augmentations: bool):
    if no_augmentations:
        return transforms.Compose([Numpy2Torch()])

    transformations = []

    # cropping
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'none':
        transformations.append(UniformCrop(cfg.AUGMENTATION.CROP_SIZE))
    else:
        transformations.append(ImportanceRandomCrop(cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_SHIFT:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        img_s1, img_s2, label = args
        img_s1_tensor = TF.to_tensor(img_s1)
        img_s2_tensor = TF.to_tensor(img_s2)
        label_tensor = TF.to_tensor(label)
        return img_s1_tensor, img_s2_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img_s1, img_s2, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img_s1 = np.flip(img_s1, axis=1)
            img_s2 = np.flip(img_s2, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img_s1 = np.flip(img_s1, axis=0)
            img_s2 = np.flip(img_s2, axis=0)
            label = np.flip(label, axis=0)

        img_s1 = img_s1.copy()
        img_s2 = img_s2.copy()
        label = label.copy()

        return img_s1, img_s2, label


class RandomRotate(object):
    def __call__(self, args):
        img_s1, img_s2, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img_s1 = np.rot90(img_s1, k, axes=(0, 1)).copy()
        img_s2 = np.rot90(img_s2, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img_s1, img_s2, label


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img_s1, img_s2, label = args
        if img_s2 is not None:
            rescale_factors = np.random.uniform(self.min_factor, self.max_factor, img_s2.shape[-1])
            img_s2 = np.clip(img_s2 * rescale_factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_s1, img_s2, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        img_s1, img_s2, label = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img_s2.shape[-1])
        img_s2 = np.clip(np.power(img_s2, gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_s1, img_s2, label


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, args):
        img_s1, img_s2, label = args
        height, width, _ = label.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        img_s1 = img_s1[y:y+self.crop_size, x:x+self.crop_size, ]
        img_s2 = img_s2[y:y + self.crop_size, x:x + self.crop_size, ]
        label = label[y:y+self.crop_size, x:x+self.crop_size, ]
        return img_s1, img_s2, label

    def __call__(self, args):
        img_s1, img_s2, label = self.random_crop(args)
        return img_s1, img_s2, label


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):

        sample_size = 20
        balancing_factor = 5

        random_crops = [self.random_crop(args) for _ in range(sample_size)]
        crop_weights = np.array([crop_label.sum() for _, _, crop_label in random_crops]) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img_s1, img_s2, label = random_crops[sample_idx]

        return img_s1, img_s2, label
