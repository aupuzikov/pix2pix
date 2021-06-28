import matplotlib.pyplot as plt


def save_image(tensor, name='result.png'):
    inp = tensor
    inp = inp.numpy().transpose((0, 2, 3, 1))

    for j, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
        inp[:, :, j] = inp[:, :, j] * std + mean

    plt.rcParams["figure.figsize"] = (25, 25)
    plt.imsave(name, inp)


def save_image_batch(tensor, rows=5, wspace=0, hspace=-0.3, name='default_name.png'):
    inp = tensor
    #     print(inp.shape)
    #     inp = torch.from_numpy(inp.numpy().transpose((0,3, 1, 2)))

    inp = inp.numpy().transpose((0, 2, 3, 1))
    count = len(inp)

    plt.rcParams["figure.figsize"] = (25, 25)
    # plt.show(block=True)

    plt.figure(figsize=(15, 16))

    import math
    cols = math.ceil(len(tensor) / rows)
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    #     fig.tight_layout()
    for i, image in enumerate(inp):
        for j, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
            #             print(i)
            image[:, :, j] = image[:, :, j] * std + mean
        #         import pdb; pdb.set_trace()
        axes[i // cols, i % cols].axis("off")
        axes[i // cols, i % cols].imshow(image)
    fig.savefig(name)
