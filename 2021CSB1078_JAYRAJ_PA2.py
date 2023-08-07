import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

def find_index(pt,points):
    ind=0
    for point in points:
        if pt==point:
            return ind
        ind+=1

def delaunay_morph(img1,img2,factor,landmark_points_img1,landmark_points_img2,triangle_ind):
    morph_image=np.zeros_like(img1)
    landmark_points_morph=[]
    for i in range(len(landmark_points_img1)):
        temp1=int((1-factor)*(landmark_points_img1[i][0])+(factor)*(landmark_points_img2[i][0]))
        temp2=int((1-factor)*(landmark_points_img1[i][1])+(factor)*(landmark_points_img2[i][1]))
        landmark_points_morph.append((temp1,temp2))
    
    for t in triangle_ind:
        img1_pt1=landmark_points_img1[t[0]]
        img1_pt2=landmark_points_img1[t[1]]
        img1_pt3=landmark_points_img1[t[2]]

        morph_pt1=landmark_points_morph[t[0]]
        morph_pt2=landmark_points_morph[t[1]]
        morph_pt3=landmark_points_morph[t[2]]

        img2_pt1=landmark_points_img2[t[0]]
        img2_pt2=landmark_points_img2[t[1]]
        img2_pt3=landmark_points_img2[t[2]]

        triangle1=np.array([img1_pt1,img1_pt2,img1_pt3],np.int32)
        x,y,w,h=cv2.boundingRect(triangle1)
        rel_img1_pt1=(img1_pt1[0]-x,img1_pt1[1]-y)
        rel_img1_pt2=(img1_pt2[0]-x,img1_pt2[1]-y)
        rel_img1_pt3=(img1_pt3[0]-x,img1_pt3[1]-y)
        cropped_triangle1=img1[y:y+h,x:x+w]

        cropped_image1_mask=np.zeros((h,w),img1.dtype)
        
        points=np.array([[img1_pt1[0]-x,img1_pt1[1]-y],
                         [img1_pt2[0]-x,img1_pt2[1]-y],
                         [img1_pt3[0]-x,img1_pt3[1]-y]
                         ])
        cv2.fillConvexPoly(cropped_image1_mask,points,255)
       

        cropped_triangle1=cv2.bitwise_and(cropped_triangle1,cropped_triangle1,mask=cropped_image1_mask)

        triangle2=np.array([img2_pt1,img2_pt2,img2_pt3],np.int32)
        x,y,w,h=cv2.boundingRect(triangle2)
        rel_img2_pt1=(img2_pt1[0]-x,img2_pt1[1]-y)
        rel_img2_pt2=(img2_pt2[0]-x,img2_pt2[1]-y)
        rel_img2_pt3=(img2_pt3[0]-x,img2_pt3[1]-y)
        cropped_triangle2=img2[y:y+h,x:x+w]
        # print(cropped_triangle2)
        cropped_image2_mask=np.zeros((h,w),img2.dtype)
        # cv2.imshow("before",cropped_triangle2)
        points=np.array([[img2_pt1[0]-x,img2_pt1[1]-y],
                         [img2_pt2[0]-x,img2_pt2[1]-y],
                         [img2_pt3[0]-x,img2_pt3[1]-y]
                         ])
        cv2.fillConvexPoly(cropped_image2_mask,points,255)
        cropped_triangle2=cv2.bitwise_and(cropped_triangle2,cropped_triangle2,mask=cropped_image2_mask)
        
        morph_triangle=np.array([morph_pt1,morph_pt2,morph_pt3],np.int32)
        x,y,w,h=cv2.boundingRect(morph_triangle)
        rel_morph_pt1=(morph_pt1[0]-x,morph_pt1[1]-y)
        rel_morph_pt2=(morph_pt2[0]-x,morph_pt2[1]-y)
        rel_morph_pt3=(morph_pt3[0]-x,morph_pt3[1]-y)
        p_of_img1=np.float32([rel_img1_pt1,rel_img1_pt2,rel_img1_pt3])
        p_of_morph=np.float32([rel_morph_pt1,rel_morph_pt2,rel_morph_pt3])
        p_of_img2=np.float32([rel_img2_pt1,rel_img2_pt2,rel_img2_pt3])
        trans1=cv2.getAffineTransform(p_of_img1,p_of_morph)
        trans2=cv2.getAffineTransform(p_of_img2,p_of_morph)

        morph_part1=cv2.warpAffine(cropped_triangle1,trans1,(w,h),flags=cv2.INTER_NEAREST)
        morph_part2=cv2.warpAffine(cropped_triangle2,trans2,(w,h),flags=cv2.INTER_NEAREST)
        background=morph_image[y:y+h,x:x+w]
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        _, mask_triangles_designed = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        morph_part1 = cv2.bitwise_and(morph_part1, morph_part1, mask=mask_triangles_designed)
        morph_part2 = cv2.bitwise_and(morph_part2, morph_part2, mask=mask_triangles_designed)

        final_triangle=cv2.addWeighted(morph_part1,(1-factor),morph_part2,factor,0)
        
        background=cv2.add(background,final_triangle)
        
        morph_image[y:y+h,x:x+w]=background

    return morph_image 

def myfilter(img):
    k=5
    output=np.copy(img)
    size=img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(7,size[0]-7):
        for j in range(7,size[1]-7):
            if(gray[i][j]>=200):
                output[i][j]=cv2.medianBlur(img[i-2:i+2,j-2:j+2],k)[0][0]
    return output

                
  
img1=cv2.imread("img1.png",cv2.IMREAD_COLOR)
img2=cv2.imread("img2.png",cv2.IMREAD_COLOR)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

landmark_points1=[]
landmark_points2=[]

print("Enter your choice")
print("Press 0 for inbuilt landmark point detection")
print("Press 1 for reading tie points from file")
choice=int(input("Your Choice :"))
if(choice==1):
    with open("landmark.txt") as file:
        line=file.readline()
        while (len(line)!=0):
            values=line.split()
            assert(len(values)==4)
            landmark_points1.append((int(values[0]),int(values[1])))
            landmark_points2.append((int(values[2]),int(values[3])))
            line=file.readline()

else:  
    face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = face_detector(gray1)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray1, face)
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmark_points1.append((x,y))
        break

    faces2 = face_detector(gray2)
    for face in faces2:
        face_landmarks = dlib_facelandmark(gray2, face)
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            landmark_points2.append((x,y))
        break




h,w=gray1.shape
landmark_points1.extend([(0,0),(w-1,0),(0,h-1),(w-1,h-1)])        
np_points1=np.array(landmark_points1,np.int32)

h,w=gray1.shape
rect=(0,0,w,h)
subdiv=cv2.Subdiv2D(rect)
subdiv.insert(landmark_points1)
triangles1=subdiv.getTriangleList()
triangles1=np.array(triangles1,dtype=np.int32)
triangle_ind=[]
for t in triangles1:
    ind=[]
    pt1=(t[0],t[1])
    pt2=(t[2],t[3])
    pt3=(t[4],t[5])
    ind.append(find_index(pt1,landmark_points1))
    ind.append(find_index(pt2,landmark_points1))
    ind.append(find_index(pt3,landmark_points1))
    triangle_ind.append(ind)


h,w=gray2.shape
landmark_points2.extend([(0,0),(w-1,0),(0,h-1),(w-1,h-1)])        
np_points2=np.array(landmark_points2,np.int32)

triangles2=[]
for t in triangle_ind:
    ls=[]
    ls.extend(landmark_points2[t[0]])
    ls.extend(landmark_points2[t[1]])
    ls.extend(landmark_points2[t[2]])
    triangles2.append(ls)

# output=delaunay_morph(img1,img2,0.8,np_points1,np_points2,triangle_ind)

images=[]
frames=100
for i in range(frames+1):
    output=delaunay_morph(img1,img2,i/frames,np_points1,np_points2,triangle_ind)
    output = cv2.medianBlur(output,5)
    images.append(output)
height,width,layers=images[0].shape
size=(width,height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('korean.mp4',fourcc, 35, size)
for i in range(len(images)):
    out.write(images[i])
out.release()


