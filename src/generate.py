import argparse

import PIL
import torch
from torchvision.transforms import transforms

from model import generator
from utils import utils


def main(model_path, image_path, result_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    from PIL import Image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    # print(image.shape)
    g = generator.Generator()
    g.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        generated_image = g(image)
        utils.save_image(generated_image, name=result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", required=True, help="path to model")
    parser.add_argument("--image_path", required=True, help="path to image with edges")
    parser.add_argument("--result_path", required=False, default='./data/result.png',
                        help="result image path")

    args = parser.parse_args()
    main(model_path=args.model_path, image_path=args.image_path, result_path=args.result_path)
