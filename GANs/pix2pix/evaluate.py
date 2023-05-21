import pytorch_lightning as pl
import glob
from codes.pixTopix import pixToPix
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from pandas.core.common import flatten
from tqdm import tqdm


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

out_path = 'out/'
in_path = 'D:\\Datasets\\seg_mask_b_gen'
img_path_list = []

model = pixToPix().load_from_checkpoint(
    'D:/MachineLearningCollection/GANs/pix2pix/ckpt/FNAC/pix2pix_epoch=379-GLoss=0.00-DLoss=0.00_FNAC.ckpt').to('cuda')

for data_pth in glob.glob(in_path + '\\*'):
    img_path_list.append(glob.glob(data_pth + '\\*'))

img_path_list = list(flatten(img_path_list))
img_path_list = [a.replace('\\', '/') for a in img_path_list]


for img_pth in tqdm(img_path_list):
    img = pil_loader(img_pth)
    label = img_pth.split('/')[-2]
    file_name = img_pth.split('/')[-1]
    file_name = file_name.split('.')[0]
    img_tensor = transform(img).to('cuda')
    img_tensor = img_tensor.reshape(
        1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
    print(img_tensor.shape)

    out_img = model.gen(img_tensor)
    save_image(out_img, f"{out_path}{label}/{file_name}.jpg")
