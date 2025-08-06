# from https://github.com/AlienCat-K/3D-IoU-Python/blob/master/3D-IoU-Python.py
# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def box3d_iou_orthogonal(xyzlwh1, xyzlwh2):
    """
    Compute 3D bounding box IoU. 
    Assume the boxes are aligned with the axis.
    Don't use triangles/convex hulls, they are *very slow*.
    """
    # normalize sizes
    xyzlwh1[..., 3:] = np.abs(xyzlwh1[..., 3:])
    xyzlwh2[..., 3:] = np.abs(xyzlwh2[..., 3:])

    x1, y1, z1, l1, w1, h1 = xyzlwh1
    x2, y2, z2, l2, w2, h2 = xyzlwh2

    def box3d_vol_orthogonal(l, w, h):
        return l*w*h

    def overlap_1d(x1, x2, y1, y2): 
        """
        return the overlap of 1d segment [x1, x2] and [y1, y2]
        """
        return max(0, min(x2, y2) - max(x1, y1))
    
    vol1 = box3d_vol_orthogonal(l1, w1, h1)
    vol2 = box3d_vol_orthogonal(l2, w2, h2)


    overlap_x = overlap_1d(x1 - l1/2, x1 + l1/2, x2 - l2/2, x2 + l2/2)
    overlap_y = overlap_1d(y1 - w1/2, y1 + w1/2, y2 - w2/2, y2 + w2/2)
    overlap_z = overlap_1d(z1 - h1/2, z1 + h1/2, z2 - h2/2, z2 + h2/2)

    inter_vol = box3d_vol_orthogonal(overlap_x, overlap_y, overlap_z)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    return iou

def batch_box3d_iou_orthogonal(xyzlwh1, xyzlwh2):
    """
    xyzlwh1: (N, 6) [x, y, z, l, w, h]
    xyzlwh2: (N, 6) [x, y, z, l, w, h]
    Compute 3D bounding box IoU. 
    Assume the boxes are aligned with the axis.
    Don't use triangles/convex hulls, they are *very slow*.
    """
    # normalize sizes
    xyzlwh1[..., 3:] = np.abs(xyzlwh1[..., 3:])
    xyzlwh2[..., 3:] = np.abs(xyzlwh2[..., 3:])

    xyzlwh1 = np.transpose(xyzlwh1) # (6, N)
    xyzlwh2 = np.transpose(xyzlwh2) # (6, N)

    x1, y1, z1, l1, w1, h1 = xyzlwh1
    x2, y2, z2, l2, w2, h2 = xyzlwh2 # each is (N)

    def box3d_vol_orthogonal(l, w, h):
        return l*w*h

    def overlap_1d(x1, x2, y1, y2): 
        """
        return the overlap of 1d segment [x1, x2] and [y1, y2]
        """
        return np.maximum(0, np.minimum(x2, y2) - np.maximum(x1, y1))
    
    vol1 = box3d_vol_orthogonal(l1, w1, h1)
    vol2 = box3d_vol_orthogonal(l2, w2, h2)


    overlap_x = overlap_1d(x1 - l1/2, x1 + l1/2, x2 - l2/2, x2 + l2/2)
    overlap_y = overlap_1d(y1 - w1/2, y1 + w1/2, y2 - w2/2, y2 + w2/2)
    overlap_z = overlap_1d(z1 - h1/2, z1 + h1/2, z2 - h2/2, z2 + h2/2)

    inter_vol = box3d_vol_orthogonal(overlap_x, overlap_y, overlap_z)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    return iou # (N)

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    
        NOTE: x ~ length, y ~ height, z ~ width!!!
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size # NOTE: input has to be (x, y, z, size_x, size_z, size_y), weird!!!
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_3d_box_normal(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    
        NOTE: this time, the dimensions are 0-3, 1-4, 2-5
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    # l,w,h = box_size
    l, h, w = box_size # NOTE: thus, the input can be (x, y, z, size_x, size_y, size_z)
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_minmax_corners(corners_3d):
    """
    get the min and max corners of the 3d box
    """
    min_corners = np.min(corners_3d, axis=0)
    max_corners = np.max(corners_3d, axis=0)
    # return min_corners, max_corners
    return np.concatenate([min_corners, max_corners])

def batch_get_minmax_corners(corners_3d):
    """
    get the min and max corners of the 3d box
    """
    min_corners = np.min(corners_3d, axis=1) # [N, 8, 3] => [N, 3]
    max_corners = np.max(corners_3d, axis=1)
    return np.concatenate([min_corners, max_corners], axis=1)

def from_minmax_to_corners(corners_minmax):
    """
    get the corners of the 3d box from min and max corners
    """
    center = (corners_minmax[0:3] + corners_minmax[3:6]) / 2
    size = corners_minmax[3:6] - corners_minmax[0:3]
    corners_3d = get_3d_box(size, 0, center)
    return corners_3d

def from_minmax_to_corners_normal(corners_minmax):
    """
    get the corners of the 3d box from min and max corners
    """
    center = (corners_minmax[0:3] + corners_minmax[3:6]) / 2
    size = corners_minmax[3:6] - corners_minmax[0:3]
    corners_3d = get_3d_box_normal(size, 0, center)
    return corners_3d

def from_minmax_to_xyzhwl(corners_minmax):
    """
    get the center and size of the 3d box from min and max corners
    """
    center = (corners_minmax[0:3] + corners_minmax[3:6]) / 2
    size = corners_minmax[3:6] - corners_minmax[0:3]
    return np.concatenate([center, size])

def batch_from_minmax_to_xyzhwl(corners_minmax):
    """
    get the center and size of the 3d box from min and max corners
    """
    center = (corners_minmax[:, 0:3] + corners_minmax[:, 3:6]) / 2
    size = corners_minmax[:, 3:6] - corners_minmax[:, 0:3]
    return np.concatenate([center, size], axis=1)

if __name__=='__main__':
    print('------------------')
    # get_3d_box(box_size, heading_angle, center)
    corners_3d_ground  = get_3d_box((1.497255,1.644981, 3.628938), -1.531692, (2.882992 ,1.698800 ,20.785644)) 
    corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280 ))
    (IOU_3d,IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
    print (IOU_3d,IOU_2d) #3d IoU/ 2d IoU of BEV(bird eye's view)
      