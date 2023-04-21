from typing import Tuple
import os
import itertools


from torch.utils.data import Dataset, random_split
import numpy as np

from PIL import Image
from PIL import ImageDraw
from enum import Enum


class Shape(Enum):
    rectangle = 0
    circle = 1
    # regular_polygon pentagon


class Color(Enum):
    red = 0
    blue = 1
    yellow = 2


class Object:
    def __init__(
        self,
        shape: Shape,
        color: Color,
        coordinates: Tuple,
        rotation: float = 1,
        scale: float = 1,
    ):
        """
        Args:
            shape: Shape
            color: Color
            coordinates: (x, y) between (0, 1)
            rotation: angle between (0, 360)
            scale:
        """
        self.shape = shape
        self.color = color
        self.coordinates = coordinates
        self.rotation = rotation
        self.scale = scale
        self.size = 0.25


def draw_object(draw, object: Object):
    width, height = draw.im.size
    x, y = object.coordinates[0] * width, object.coordinates[0] * height
    r = object.size * object.scale * min(width, height)
    if object.shape == Shape.rectangle:
        draw.regular_polygon(((x, y), r), n_sides=4, fill=object.color.name, rotation=object.rotation)
    elif object.shape == Shape.circle:
        xy = [(x + r, y + r), (x - r, y - r)]
        draw.ellipse(xy, fill=object.color.name)


def draw(object, size=(64,64)):
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    draw_object(draw, object)
    return image


def generate(file_path, size=(64, 64)):
    # total: 2*3*16*16*18*15
    # for debug: rotation only 3 values
    shape_range = [s.value for s in Shape]
    color_range = [c.value for c in Color]
    coordinate_range = [x / 64 for x in range(0, 64, 4)]
    rotation_range = [0, 45, 90] # list(range(0, 90, 5))
    scale_range = [s / 10 for s in range(5, 20)]
    latents = np.array(
        list(
            itertools.product(
                shape_range,
                color_range,
                coordinate_range,
                coordinate_range,
                rotation_range,
                scale_range,
            )
        )
    )
    images = [
        np.array(
            draw(Object(Shape(s), Color(c), (x, y), rotation, scale), size)
        )
        for s, c, x, y, rotation, scale in latents
    ]
    images = np.stack(images)
    np.savez(file_path, images=images, latents=latents)

    # random.seed(100)
    # target = Object(
    #     shape=random.choice(list(Shape)),
    #     color=random.choice(list(Color)),
    #     coordinates=(random.uniform(0, 1), random.uniform(0, 1)),
    #     rotation=360*random.gauss(0, 1),
    #     scale=1 + random.gauss(0, 1)
    # )


class GeneratedsDataset(Dataset):
    def __init__(self, root="data", transform=None):
        file_path = os.path.join(root, "Generateds.npz")
        if not os.path.exists(file_path):  # isfile(fname)
            generate(file_path, size=(64, 64))
        data = np.load(file_path, allow_pickle=True)
        self.images = data["images"]
        self.latents = data["latents"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        latent = self.latents[idx]
        if self.transform:
            image = self.transform(image)
        return image, latent


def Generateds_sets(train_rate=0.8):
    dSprites = GeneratedsDataset()
    size = len(dSprites)
    train_size = int(size * train_rate)
    val_size = size - train_size
    train_set, val_set = random_split(dSprites, [train_size, val_size])
    return train_set, val_set


image1 = draw(Object(Shape.rectangle, Color.red, (0, 0), rotation=0, scale=0.5))
image2 = draw(Object(Shape.rectangle, Color.blue, (0, 1), rotation=0, scale=1))
image3 = draw(Object(Shape.rectangle, Color.yellow, (1, 1), rotation=0, scale=2))