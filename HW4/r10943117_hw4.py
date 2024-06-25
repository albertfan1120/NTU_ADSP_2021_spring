import cv2


def SSIM(A, B, c1, c2):
    L = 255
    
    mu_x, mu_y = A.mean(), B.mean()
    var_x, var_y = ((A - mu_x) ** 2).mean(), ((B - mu_y) ** 2).mean()
    cor_xy = ((A - mu_x) * (B - mu_y)).mean()
    
    result = (2*mu_x*mu_y + (c1*L)**2) * (2*cor_xy + (c2*L)**2) / \
             ((mu_x**2 + mu_y**2 + (c1*L)**2) * (var_x + var_y + (c2*L)**2))

    return result


if __name__ == '__main__':
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    c1 = c2 = 1 / (255 ** 0.5)
    
    result = SSIM(img1, img2, c1, c2)
    
    print("SSIM between img1 and img2 is {:.3f}".format(result))