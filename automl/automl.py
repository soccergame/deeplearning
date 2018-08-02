# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('messi.jpg')
#
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('messi.jpg')
#
# # blur = cv2.blur(img,(5,5))
# # blur =cv2.GaussianBlur(img,(5,5),0)
# blur = cv2.medianBlur(img,5)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('j.png')
# kernel = np.ones((9,9),np.uint8)
# # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# #erosion = cv2.erode(img,kernel,iterations = 1)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# plt.subplot(221),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(tophat),plt.title('tophat')
# plt.xticks([]), plt.yticks([])
# plt.show()
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('messi.jpg',0)
#
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#         ssh-keygen -t rsa -C "18811442380@163.com"
#   /home/typhoon/linqinggit/.ssh/id_rsa
# plt.show()
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('double_edge.jpg',0)
#
# # Output dtype = cv2.CV_8U
# sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
#
# # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
# sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
#
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1,3,3),plt.imshow(sobelx64f,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
#
# plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()