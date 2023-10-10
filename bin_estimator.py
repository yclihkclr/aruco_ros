#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
import numpy as np
import threading
from transform import quaternion_matrix
import transform

class BinEstimator():
    def __init__(self):
        rospy.init_node('bin_estimator', anonymous=True)
        self.sub1=rospy.Subscriber("/aruco_simple/pose", Pose, self.marker1_cb)
        self.sub2=rospy.Subscriber("/aruco_simple/pose2", Pose, self.marker2_cb)

        self.pose1= Pose()
        self.pose2= Pose()
        self.pose_1 = None
        self.pose_2 = None
        self.T1=np.eye(4)
        self.T2=np.eye(4)
        self.marker_size = 0.05
        self.marker_to_robot=np.array([[ 0,  1,  0, 0.18-0.5*self.marker_size], 
                                       [ 0,  0,  1, 0.275],
                                       [ 1,  0,  0, 0.5*self.marker_size],
                                       [ 0,  0,  0, 1]])

        #bin dimension
        self.height_gap = 0.04 # gab between bin center and mark center in Z
        self.width = 0.27
        self.length = 0.37
        self.height = 0.15

        self.m2_T_left = np.array([ [0,0,1,self.height_gap],
                            [1,0,0,0    ],
                            [0,1,0,0    ],
                            [0,0,0,1    ]])
        
        self.left_T_right=np.array([[1,0,0,0     ],
                            [0,1,0,-self.length],
                            [0,0,1,0     ],
                            [0,0,0,1     ]])
        
        self.left_T_front=np.array([[1,0,0,self.width/2],
                            [0,1,0,-self.length/2],
                            [0,0,1,0     ],
                            [0,0,0,1    ]])
        
        self.left_T_back=np.array([ [1,0,0,-self.width/2],
                            [0,1,0,-self.length/2],
                            [0,0,1,0     ],
                            [0,0,0,1     ]])

        self.left_T_center=np.array([ [1,0,0,0],
                                [0,1,0,-self.length/2],
                                [0,0,1,0     ],
                                [0,0,0,1     ]])

        self.compute_inv_quatpose(self.marker_to_robot)
        spin_thread = threading.Thread(target=rospy.spin)
        spin_thread.start() 

    def compute_inv_quatpose(self,markerT1):
        inv_quatpose = self.Tpose_to_quatpose(np.linalg.inv(self.marker_to_robot))
        print("inv_quatpose is :",inv_quatpose)
        return inv_quatpose

    def marker1_cb(self,data):
        self.pose1 = data
        # print("pose1: ",pose1)
        self.pose_1=np.array([self.pose1.position.x, self.pose1.position.y, self.pose1.position.z, self.pose1.orientation.x, self.pose1.orientation.y,self.pose1.orientation.z,self.pose1.orientation.w])
        # print("pose1: ",self.pose_1)

    def marker2_cb(self,data):
        self.pose2 = data
        # print("pose2: ",pose2)
        self.pose_2=np.array([self.pose2.position.x, self.pose2.position.y, self.pose2.position.z, self.pose2.orientation.x, self.pose2.orientation.y,self.pose2.orientation.z,self.pose2.orientation.w])
        # print("pose2: ",self.pose_2)

    def quatpose_to_Tpose(self,quatpose):
        Tpose = transform.quaternion_matrix(quatpose[-4:])
        Tpose[:3,3] = quatpose[:3]
        return Tpose
    
    def Tpose_to_quatpose(self,Tpose):
        pose_tran = Tpose[:3,3].tolist()
        pose_quat = transform.quaternion_from_matrix(Tpose).tolist()
        quatpose = pose_tran+pose_quat
        return quatpose
    
    def estimate_bin(self,pose1,pose2):
        rTm1 = self.marker_to_robot

        cTm1 = self.quatpose_to_Tpose(pose1)
        rTc=np.dot(rTm1,np.linalg.inv(cTm1))
        
        cTm2 = self.quatpose_to_Tpose(pose2)
        rTm2 = np.dot(rTc,cTm2)
        # print("rTm2 is: ",self.Tpose_to_quatpose(rTm2))

        rTw0=np.dot(rTm2,self.m2_T_left)
        rTw1=np.dot(rTw0,self.left_T_right)
        rTw2=np.dot(rTw0,self.left_T_front)
        rTw3=np.dot(rTw0,self.left_T_back)
        rTwc=np.dot(rTw0,self.left_T_center)
        pose_w0 = self.Tpose_to_quatpose(rTw0)
        pose_w1 = self.Tpose_to_quatpose(rTw1)
        pose_w2 = self.Tpose_to_quatpose(rTw2)
        pose_w3 = self.Tpose_to_quatpose(rTw3)
        pose_wc = self.Tpose_to_quatpose(rTwc)
        return pose_w0,pose_w1,pose_w2,pose_w3,pose_wc


if __name__ == '__main__':

    test = BinEstimator()
    # print(test.pose_1)
    while True:
        if test.pose_1 is not None and test.pose_2 is not None:
            pose_w0,pose_w1,pose_w2,pose_w3,pose_wc=test.estimate_bin(test.pose_1,test.pose_2)
            # print(f"Left {pose_w0}, Right {pose_w1}, Front {pose_w2}, Back {pose_w3}, Center {pose_wc}\n")

   