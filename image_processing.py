from skimage import io, measure
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu
from skimage.transform import rescale

import matplotlib.pyplot as plt

image = io.imread('images/telugu_01.jpg', as_gray=True)

# Denoising
denoised_image = denoise_tv_chambolle(image, weight=0.1, multichannel=True)

# Thresholding
threshold = threshold_otsu(denoised_image)
thresholded_image = denoised_image > threshold

# Scaling
rescaled_image = rescale(thresholded_image, 0.75)

# Display images
fig, axes = plt.subplots(ncols=4, figsize= (8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3)
ax[3] = plt.subplot(1, 4, 4)

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(denoised_image, cmap=plt.cm.gray)
ax[1].set_title('Denoised')
ax[1].axis('off')

ax[2].imshow(thresholded_image, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

ax[3].imshow(rescaled_image, cmap=plt.cm.gray)
ax[3].set_title('Rescaled')
ax[3].axis('off')

plt.show()