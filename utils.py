import torch
from PIL import Image
import os
import glob
from model import Encoder, Decoder, Discriminator
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import logging
import datetime
from torch.utils.data import Dataset


class PngDataset(Dataset):
    def __init__(self, png_path_list, message):

        self.png_path_list = png_path_list
        self.message = message

        self.transform = transforms.Compose([
                        transforms.Resize([64,64]),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        ])

    def __len__(self):
        return len(self.png_path_list)

    def __getitem__(self, idx):

        png_tensor = Image.open(self.png_path_list[idx])
        png_tensor = self.transform(png_tensor)

        png_message = torch.tensor(self.message)

        return png_tensor, png_message



def setup_logger(prefix='model_training', log_dir='logs', console_level=logging.ERROR):
    """
    初始化并配置日志器，返回一个已经配置好的logger实例。

    参数:
        prefix (str): 日志文件名的前缀。
        log_dir (str): 存放日志文件的目录，默认为'logs'。
        console_level (logging.LEVEL): 控制台日志级别，默认只显示错误及以上级别的信息。

    返回:
        logger: 配置好的logging.Logger实例。
    """

    def create_log_filename():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{prefix}_{timestamp}.log'

    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建自定义日志文件名
    log_file_name = create_log_filename()
    log_path = os.path.join(log_dir, log_file_name)

    # 创建一个logger
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入自定义日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建一个formatter，用于设置日志格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # 添加FileHandler到logger
    logger.addHandler(file_handler)

    # 创建一个StreamHandler，用于将错误级别及以上的日志输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)

    # 添加StreamHandler到logger
    logger.addHandler(stream_handler)

    return logger



def showimg(img_tensor):
    # 将图像从(c, h, w)转换为(h, w, c)以符合matplotlib的图像格式
    for i in range(0, img_tensor.shape[0]):
        img_to_show = img_tensor[i].permute(1, 2, 0).numpy()
        plt.imshow(img_to_show)
        plt.show()



def load_images(directory, size=None, scale=None):
    images = []
    for filename in glob.glob(os.path.join(directory, '*.[pP][nN][gG]'), recursive=True):
        img = Image.open(filename).convert('L')
        if size is not None:
            img = img.resize((size, size), Image.LANCZOS)
        elif scale is not None:
            img_width, img_height = img.size
            new_width = int(img_width / scale)
            new_height = int(img_height / scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        images.append(img)
    return images



def load_model(path, device, name):
    if name == 'encoder':
        model = Encoder()
    elif name == 'decoder':
        model = Decoder()
    elif name == 'discriminator':
        model = Discriminator()
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model



def save_images(tensor, directory, name="0"):
    """
    将归一化的 (b, c, h, w) 形状的 PyTorch Tensor 存储为图像文件。

    参数：
    tensor (torch.Tensor) : 归一化后的图像数据，形状为 (b, c, h, w)。
    directory (str)       : 图像存储的目标目录。
    filename_prefix (str) : 图像文件名的前缀，默认为 "image_"。
    file_format (str)     : 图像文件格式，默认为 "png"。

    返回值：
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(tensor.shape[0]):
        img = tensor[i].detach().cpu().numpy()
        img *= 255  # 将归一化数据转换回 0-255 范围
        img = img.astype(np.uint8)  # 转换为 uint8 类型
        img = img.transpose(1, 2, 0)  # 将 (c, h, w) 转换为 (h, w, c) 方便 PIL 处理
        img = img.squeeze() # 适应于单通道

        pil_image = Image.fromarray(img, mode='L')
        file_path = os.path.join(directory, f"{name}")
        pil_image.save(file_path)



def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram



def save_model(encoder, decoder, discri, args, e='end'):
    encoder.eval().cpu()
    encoder_model_filename = "encoder_epoch_" + e + ".pth"
    encoder_model_path = os.path.join(args.save_model_dir, encoder_model_filename)
    torch.save(encoder.state_dict(), encoder_model_path)

    decoder.eval().cpu()
    decoder_model_filename = "decoder_epoch_" + e + ".pth"
    decoder_model_path = os.path.join(args.save_model_dir, decoder_model_filename)
    torch.save(decoder.state_dict(), decoder_model_path)

    discri.eval().cpu()
    discri_model_filename = "discri_epoch_" + e + ".pth"
    discri_model_path = os.path.join(args.save_model_dir, discri_model_filename)
    torch.save(discri.state_dict(), discri_model_path)



def eval_model(encoder, decoder, args, device):
    if args.eval_data is None:
        return
    encoder.eval()
    decoder.eval()
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # 数值在[0,1]
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    eval_dataset = datasets.ImageFolder(args.eval_data, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        img_total = 0
        img0_correct = 0
        img1_correct = 0
        for batch_id, (x, message) in enumerate(eval_loader):
            img_total = img_total + len(x)
            x = x.to(device)
            message = message.to(device)
            y = encoder(x, message)
            y = torch.clamp(y, min=0, max=1)
            m = decoder(y)
            probabilities = F.softmax(m, dim=1)
            _, predicted_classes = probabilities.max(dim=1)
            img1_correct = img1_correct + ((message == 1) & (message == predicted_classes)).float().sum().item()
            img0_correct = img0_correct + ((message == 0) & (message == predicted_classes)).float().sum().item()
        return img_total / 2, img1_correct, img0_correct



def perspective_img(imgs, w, rate):
    d = w * rate
    tl_x = random.uniform(-d, d)  # Top left corner, top
    tl_y = random.uniform(-d, d)  # Top left corner, left
    bl_x = random.uniform(-d, d)  # Bot left corner, bot
    bl_y = random.uniform(-d, d)  # Bot left corner, left
    tr_x = random.uniform(-d, d)  # Top right corner, top
    tr_y = random.uniform(-d, d)  # Top right corner, right
    br_x = random.uniform(-d, d)  # Bot right corner, bot
    br_y = random.uniform(-d, d)  # Bot right corner, right

    rect = np.array([
        [0, 0],
        [w, 0],
        [w, w],
        [0, w]], dtype="float32")

    dst = np.array([
        [tl_x, tl_y],
        [tr_x + w, tr_y],
        [br_x + w, br_y + w],
        [bl_x, bl_y + w]], dtype="float32")

    out = transforms.functional.perspective(imgs, rect, dst, interpolation=Image.BILINEAR, fill=1)
    out = torch.clamp(out, 0, 1)
    return out



def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.stack(torch.meshgrid(torch.arange(N_blur), torch.arange(N_blur), indexing='ij'), -1) - (.5 * (N-1))
    coords = coords.float()
    manhat = torch.sum(torch.abs(coords), -1)

    # nothing, default
    vals_nothing = (manhat < .5).float()

    # gauss
    sig_gauss = torch.rand([]) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords**2, -1) / 2.0 / sig_gauss**2)

    # line
    theta = torch.rand([]) * 2.0 * np.pi
    v = torch.tensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.rand([]) * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand([]) * (.5 * (N-1) + .1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists**2 / 2.0 / sig_line**2) * (manhat < w_line).float()

    # Select blur type based on probs
    t = torch.rand([])
    if t < probs[0]:
        vals = vals_gauss
    elif t < probs[0] + probs[1]:
        vals = vals_line
    else:
        vals = vals_nothing

    vals = vals / torch.sum(vals)

    return vals



def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    # Generate random hue adjustments
    rnd_hue = torch.rand((batch_size, 1, 1, 1), dtype=torch.float32) * (2 * rnd_hue) - rnd_hue
    # Generate random brightness adjustments
    rnd_brightness = torch.rand((batch_size, 1, 1, 1), dtype=torch.float32) * (2 * rnd_bri) - rnd_bri
    # Return the combined adjustments
    return rnd_hue + rnd_brightness



def color_manipulation(encoded_image, args, ramp_fn):
    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, encoded_image.size(0))

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat
    rnd_sat = rnd_sat.to(encoded_image.device)

    contrast_scale = torch.rand(encoded_image.size(0), device=encoded_image.device) * (
                contrast_params[1] - contrast_params[0]) + contrast_params[0]
    contrast_scale = contrast_scale.view(-1, 1, 1, 1)

    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness.to(encoded_image.device)
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # 计算亮度图
    lum_weights = torch.tensor([0.3, 0.6, 0.1], device=encoded_image.device).view(1, 3, 1, 1)
    encoded_image_lum = torch.sum(encoded_image * lum_weights, dim=1, keepdim=True)

    # 调整饱和度
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    encoded_image = torch.clamp(encoded_image, 0, 1)

    return encoded_image



def noise(encoded_image, rnd_noise):
    noise = torch.normal(mean=0.0, std=rnd_noise, size=encoded_image.size())
    encoded_image = encoded_image + noise.to(encoded_image.device)
    encoded_image = torch.clamp(encoded_image, 0, 1)

    return encoded_image



def apply_transformations(input_tensor, args, ramp_fn, H, W):
    # Resizing
    rnd_rate = torch.rand(2) * ramp_fn(args.rnd_resize_ramp) * args.rnd_resize
    new_H = random.randint(int(H * (1 - rnd_rate[0])), int(H * (1 + rnd_rate[0])))
    new_W = random.randint(int(W * (1 - rnd_rate[1])), int(W * (1 + rnd_rate[1])))
    resize_transform = transforms.Resize((new_H, new_W))
    resized_tensor = resize_transform(input_tensor)
    resized_tensor = torch.clamp(resized_tensor, 0, 1)

    rnd_trans_rate = torch.rand(2) * ramp_fn(args.rnd_trans_ramp) * args.rnd_trans
    rnd_scal_rate = torch.rand([]) * ramp_fn(args.rnd_scal_ramp) * args.rnd_scal
    angle = torch.rand([]) * ramp_fn(args.rnd_rot_ramp) * args.rnd_rot
    trans = transforms.Compose([
        transforms.RandomAffine(degrees=[-angle, angle], translate=(rnd_trans_rate[0], rnd_trans_rate[1]),
                                        scale=(1 - rnd_scal_rate, 1 + rnd_scal_rate), fill=1)]) # Rotate, translate, scale
    trans_tensor = trans(resized_tensor)

    # Final resizing to original shape (N, C, H, W)
    final_resize_transform = transforms.Resize((H, W))
    output_tensor = final_resize_transform(trans_tensor)
    output_tensor = torch.clamp(output_tensor, 0, 1)

    return output_tensor


