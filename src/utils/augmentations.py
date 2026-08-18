# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.utils.general import (
    LOGGER,
    check_version,
    colorstr,
    resample_segments,
    segment2box,
    xywhn2xyxy,
)
from src.utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(
                    height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0
                ),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

            LOGGER.info(
                prefix
                + ", ".join(
                    f"{x}".replace("always_apply=False, ", "") for x in T if x.p
                )
            )
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(
                image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0]
            )  # transformed
            im, labels = new["image"], np.array(
                [[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])]
            )
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(
        yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB
    )  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(
            random.uniform(0, w - bw)
        )  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(
                im, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            im = cv2.warpAffine(
                im, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = (
                    xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                )  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
                n, 8
            )  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            )

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10,
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(
                    im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED
                )

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = (
            [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
        )  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(
                    box, xywhn2xyxy(labels[:, 1:5], w, h)
                )  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (
                        float(jitter),
                    ) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [
                A.SmallestMaxSize(max_size=size),
                A.CenterCrop(height=size, width=size),
            ]
        T += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]  # Normalize and convert to Tensor
        LOGGER.info(
            prefix
            + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
        )
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(
            f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)"
        )
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    # Transforms to apply if albumentations not installed
    assert isinstance(
        size, int
    ), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(
        [CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (
            (math.ceil(x / self.stride) * self.stride for x in (h, w))
            if self.auto
            else self.h
        ), self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(
            im, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(
            im[top : top + m, left : left + m],
            (self.w, self.h),
            interpolation=cv2.INTER_LINEAR,
        )


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(
            im.transpose((2, 0, 1))[::-1]
        )  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


class MalariaDGAugment:
    def __init__(
        self,
        p_blur=0.5,
        p_noise=0.5,
        p_bc=0.8,
        p_color=0.5,
        p_stain=0.7,
        p_res=0.5,
        p_jpeg=0.5,
        p_distortion=0.5,
        p_vignette=0.5,
    ):
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_bc = p_bc
        self.p_color = p_color
        self.p_stain = p_stain
        self.p_res = p_res
        self.p_jpeg = p_jpeg
        self.p_distortion = p_distortion
        self.p_vignette = p_vignette

    # -------------------------
    # 1. Gaussian blur
    # -------------------------
    def gaussian_blur(self, img):
        if random.random() < self.p_blur:
            # sometimes very strong blur
            if random.random() < 0.3:
                k = random.choice([11, 13, 15])
                sigma = random.uniform(3.0, 8.0)
            else:
                k = random.choice([5, 7, 9])
                sigma = random.uniform(1.0, 3.0)

            return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

        return img

    # -------------------------
    # 2. Gaussian noise
    # -------------------------
    def gaussian_noise(self, img):
        if random.random() < self.p_noise:
            std = random.uniform(5, 20)
            noise = np.random.randn(*img.shape) * std
            img = img.astype(np.float32) + noise
            return np.clip(img, 0, 255).astype(np.uint8)
        return img

    # -------------------------
    # 3. Brightness / Contrast
    # -------------------------
    def brightness_contrast(self, img):
        if random.random() < self.p_bc:
            img = img.astype(np.float32)

            b = random.uniform(0.7, 1.3)
            c = random.uniform(0.7, 1.3)

            img = img * b
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = (img - mean) * c + mean

            return np.clip(img, 0, 255).astype(np.uint8)

        return img

    # -------------------------
    # 4. Color jitter (HSV)
    # -------------------------
    def color_jitter(self, img):
        if random.random() < self.p_color:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

            hsv[..., 0] += random.uniform(-10, 10)
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv[..., 2] *= random.uniform(0.8, 1.2)

            hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    # -------------------------
    # 5. Stain perturbation
    # -------------------------
    def stain_perturbation(self, img):
        if random.random() < self.p_stain:
            img = img.astype(np.float32)

            gains = np.random.uniform(0.8, 1.2, size=(1, 1, 3))
            img = img * gains

            return np.clip(img, 0, 255).astype(np.uint8)

        return img

    # -------------------------
    # 6. Resolution degradation
    # -------------------------
    def resolution_degradation(self, img):
        if random.random() < self.p_res:
            h, w = img.shape[:2]
            scale = random.uniform(0.3, 0.8)

            nh, nw = int(h * scale), int(w * scale)

            down = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

            return up

        return img

    # # -------------------------
    # # 7. JPEG compression
    # # -------------------------
    # def jpeg_compression(self, img):
    #     if random.random() < self.p_jpeg:
    #         quality = random.randint(30, 100)

    #         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    #         _, encimg = cv2.imencode(".jpg", img, encode_param)
    #         decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    #         return decimg

    #     return img

    # -------------------------
    # 8. Lens Distortion (Fisheye effect)
    # -------------------------
    def lens_distortion(self, img):
        if random.random() < self.p_distortion:
            h, w = img.shape[:2]

            # Camera matrix setup
            focal_length = w * random.uniform(0.9, 1.1)
            cx, cy = w / 2, h / 2
            camera_matrix = np.array(
                [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
                dtype=np.float32,
            )

            # Radial distortion coefficient (k1 < 0 mimics barrel/fisheye distortion)
            k1 = random.uniform(-0.08, -0.01)
            dist_coeffs = np.array([k1, 0, 0, 0], dtype=np.float32)

            # Map pixels and remap image
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
            )
            img = cv2.remap(
                img,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        return img

    # -------------------------
    # 9. Vignetting (Corner Darkening)
    # -------------------------
    def vignetting(self, img):
        if random.random() < self.p_vignette:
            h, w = img.shape[:2]

            # INCREASED DARKENING: Reduced the multipliers (e.g., from 0.4-0.7 down to 0.25-0.45)
            # This makes the bright center area smaller and forces the corners to get much darker.
            kernel_x = cv2.getGaussianKernel(w, random.uniform(w * 0.4, w * 0.65))
            kernel_y = cv2.getGaussianKernel(h, random.uniform(h * 0.4, h * 0.65))
            mask = kernel_y * kernel_x.T

            # Normalize the mask to ensure the center remains un-darkened
            mask = mask / mask.max()

            # Optional: Push the dark gradients even deeper into the corners
            # Shifting the baseline down simulates severe lens obstruction
            if random.random() < 0.5:
                mask = np.clip(mask * 1.2 - 0.2, 0, 1)

            # Expand dimensions to broadcast across color channels
            mask = np.expand_dims(mask, axis=-1)

            # Blend original image with darkness using the mask
            img = img.astype(np.float32) * mask
            return img.astype(np.uint8)
        return img

    # -------------------------
    # Full pipeline
    # -------------------------
    def __call__(self, img):
        """
        img: numpy array (H, W, C), uint8 RGB
        """

        img = self.gaussian_blur(img)
        img = self.gaussian_noise(img)

        img = self.brightness_contrast(img)
        img = self.color_jitter(img)

        img = self.stain_perturbation(img)

        # img = self.lens_distortion(img)
        # img = self.vignetting(img)

        # img = self.resolution_degradation(img)

        return img

class FourierDGAugment:
    """ Fourier Domain Generalization Augmentation """
    def __init__(self, image_size=640, jitter=0.1, alpha=1.0):
        self.image_size = image_size
        self.jitter = jitter
        self.alpha = alpha
        self.pre_transform=self._get_pre_transform()
        self.post_transform=self._get_post_transform()

    def _get_pre_transform(self):
        """ Get pre-processing transformations for the input image """
        img_transform = [T.ToPILImage(), T.Resize((self.image_size, self.image_size))]
        if self.jitter > 0:
            img_transform.append(T.ColorJitter(brightness=self.jitter,
                                                        contrast=self.jitter,
                                                        saturation=self.jitter,
                                                        hue=min(0.5, self.jitter)))
        img_transform += [T.RandomHorizontalFlip(), lambda x: np.asarray(x)]
        img_transform = T.Compose(img_transform)
        return img_transform

    def colorful_spectrum_mix(self, img1, img2, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, self.alpha)

        assert img1.shape == img2.shape
        h, w, c = img1.shape
        h_crop = int(h * math.sqrt(ratio))
        w_crop = int(w * math.sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(0, 1))
        img2_fft = np.fft.fft2(img2, axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                            h_start:h_start + h_crop,
                                                                                            w_start:w_start + w_crop]
        img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                            h_start:h_start + h_crop,
                                                                                            w_start:w_start + w_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

        img21 = img1_abs * (np.e ** (1j * img1_pha))
        img12 = img2_abs * (np.e ** (1j * img2_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
        img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
        img21 = np.uint8(np.clip(img21, 0, 255))
        img12 = np.uint8(np.clip(img12, 0, 255))

        return img21, img12
    
    def _get_post_transform(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """ Get post image transformations. """
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return img_transform

    def __call__(self, img1, img2):
        img1, img2 = self.pre_transform(img1), self.pre_transform(img2)
        img21, img12 = self.colorful_spectrum_mix(img1, img2)
        # img21, img12 = self.post_transform(img21), self.post_transform(img12)
        return img21, img12