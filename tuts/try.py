import cv2

img = cv2.imread('tulip1.jpg',0)
width, length = img.shape
# Define corners for matrix of pixels 
# X axis c1:c2
# Y axis c3:c4
res = img
# c1, c2, c3, c4 = 0, 2, 0, 2
# posx = [1,-1,0,0,1,1,-1,-1]
# posy = [0,0,1,-1,1,-1,1,-1]
# while(1):
#     roi = img[c1:c2, c3:c4]
#     px = (c1+c2)/2
#     py = (c3+c4)/2
#     px = int(px)
#     py = int(py)
#     dir = 0
#     while(dir<8):
#         tx = px + posx[dir]
#         ty = py + posy[dir]
#         tx = int(tx)
#         ty = int(ty)
#         diff = img[tx,ty] - img[px, py]
#         if(diff<img[px,py]):
#             if(diff<40):
#                 diff = 0
#             if(diff>70):
#                 diff = 255
#             else:
#                 diff = 100
#             res[px,py] = abs(diff)
#         dir = dir+1
#     c1+=1
#     c2+=1
#     if(c2==width):
#         c1=0
#         c2=2
#         c3+=1
#         c4+=1
#         if(c4==length):
#             break

# res = cv2.resize(res, (0,0), fx=20.0, fy= 20.0)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret3)
res = cv2.Canny(img, int(ret3/2), ret3)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()




