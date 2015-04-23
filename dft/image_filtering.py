import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageOps

def low_pass(l, freq_dom, dft) :
    low_spectrum = np.zeros(freq_dom.shape)
    low_spectrum[0:l, 0:l] = np.copy(freq_dom[0:l, 0:l])
    return to_img_domain(low_spectrum, dft)

def high_pass(h, freq_dom, dft) :
    n = freq_dom.shape[0]
    high_spectrum = np.copy(freq_dom)
    high_spectrum[0:n-h, 0:n-h] = np.zeros((n, n))[0:n-h, 0:n-h]
    return to_img_domain(high_spectrum, dft)

def get_dft_vandermonde(n) :
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    cos_k = np.arange(0, n//2 + 1, dtype=np.float64)
    sin_k = np.arange(1, n//2 + 1, dtype=np.float64)
    dft = np.zeros((n,n))
    dft[:, ::2] = np.cos(cos_k*x[:, np.newaxis])
    dft[:, 1::2] = np.sin(sin_k*x[:, np.newaxis])
    return dft

def to_freq_domain(img, dft) :
    n = img.shape[0]
    freq_dom = np.zeros((n, n))
    dft_inv = np.linalg.inv(dft)
    for i in range(n) :
        freq_dom[i] = dft_inv.dot(img[i].T).T
    for i in range(n) :
        freq_dom[:, i] = dft_inv.dot(freq_dom[:, i])
    return freq_dom

def to_img_domain(freq_dom, dft) :
    n = freq_dom.shape[0]
    img_dom = np.zeros((n, n))
    for i in range(n) :
        img_dom[i] = dft.dot(freq_dom[i].T).T
    for i in range(n) :
        img_dom[:, i] = dft.dot(img_dom[:, i])
    return img_dom

bio = lambda image_path: BytesIO(data_files[image_path])
image_path = "image.jpg"
temp = np.array(ImageOps.grayscale(Image.open(image_path)))[:-1, :-1]
low_dim = min(temp.shape[0], temp.shape[1])

image = temp[:low_dim, :low_dim]
plt.figure(1)
plt.imshow(image, cmap = "gray")
plt.savefig("square_image.jpg")

dft = get_dft_vandermonde(image.shape[0])

freq_domain = to_freq_domain(image, dft)
plt.figure(2)
plt.imshow(freq_domain, cmap="gray")
plt.savefig("freq_dom.jpg")

low_passed = low_pass(image.shape[0]//3, freq_domain, dft)
plt.figure(3)
plt.imshow(low_passed, cmap="gray")
plt.savefig("low_passed.jpg")

high_passed = high_pass(2*image.shape[0]//3, freq_domain, dft)
plt.figure(4)
plt.imshow(high_passed, cmap="gray")
plt.savefig("high_passed.jpg")