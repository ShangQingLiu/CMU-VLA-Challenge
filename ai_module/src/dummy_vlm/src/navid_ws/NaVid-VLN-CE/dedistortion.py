import rospy
import sys
import numpy as np
import os
import rospkg
import cv2
import torch
import time
import json
import logging
from argparse import ArgumentParser
import yaml
from collections import deque, defaultdict
from types import SimpleNamespace
import argparse
import gzip

import imageio
from PIL import Image
from scipy.ndimage import map_coordinates

import shutil
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is not None:
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
else:
    raise RuntimeError("ffmpeg not found, please install it and add to PATH.")

from tqdm import tqdm

def map_to_sphere(x, y, z, yaw_radian, pitch_radian):


    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    # Apply rotation transformations here
    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) +
                            np.cos(theta) * np.cos(pitch_radian))

    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) -
                           np.cos(theta) * np.sin(pitch_radian),
                           np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)

    return theta_prime.flatten(), phi_prime.flatten()


def interpolate_color(coords, img, method='bilinear'):
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)


def panorama_to_plane(panorama, FOV, output_size, yaw, pitch):
    """
    panorama: 可以是
      - np.ndarray（来自 cv_bridge，通常是 BGR）
      - PIL.Image.Image
      - str 或 Path（文件路径）
    返回：PIL.Image（RGB）
    """
    # 1) 读入/标准化为 RGB 的 np.ndarray
    if isinstance(panorama, np.ndarray):
        if panorama.ndim == 3 and panorama.shape[2] == 3:
            pano_array = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)  # BGR->RGB
        else:
            pano_array = panorama
        pano_h, pano_w = pano_array.shape[:2]

    elif isinstance(panorama, Image.Image):
        pano_array = np.array(panorama.convert('RGB'))
        pano_w, pano_h = panorama.size

    elif isinstance(panorama, (str, os.PathLike)):
        im = Image.open(panorama).convert('RGB')
        pano_array = np.array(im)
        pano_w, pano_h = im.size

    else:
        raise TypeError("panorama must be ndarray, PIL.Image, or path string")

    # 2) 相机与网格
    W, H = output_size
    f = 0.5 * W / np.tan(np.radians(FOV) / 2.0)

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    x = u - W / 2.0
    y = H / 2.0 - v
    z = f  # 标量可广播

    yaw_r = np.radians(yaw)
    pitch_r = np.radians(pitch)

    # 3) 球面映射
    theta, phi = map_to_sphere(x, y, z, yaw_r, pitch_r)
    # 角度规范化，防止越界
    phi   = np.mod(phi, 2*np.pi)
    theta = np.clip(theta, 0.0, np.pi)

    U = phi   * pano_w / (2.0 * np.pi)
    V = theta * pano_h / np.pi

    coords = np.vstack((V.ravel(), U.ravel()))
    colors = interpolate_color(coords, pano_array, method='bilinear')
    out = colors.reshape((H, W, 3)).astype('uint8')

    return Image.fromarray(out, 'RGB')