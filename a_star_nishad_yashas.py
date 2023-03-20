import numpy as np
import pygame
import cv2
import time

from math import cos,sin
from queue import PriorityQueue 




def line_check(pt_a, pt_b, pt_center, check_pt):

    """
    Checks if check_pt point is on the same side as pt_center for a line passing through a and b.
    If point is outside line, checks that given point is 5mm away from obstacle.
    Returns True if invalid
    """
    #Unroll points on line segment AB
    pt_b_x, pt_b_y = pt_b
    pt_a_x, pt_a_y = pt_a

    #Get slope of line AB
    m_y = pt_b_y - pt_a_y
    m_x = pt_b_x - pt_a_x

    #Get constant
    c = (m_y*pt_a_x) + (m_x*pt_a_y)

    #Line equation
    line_arr = [m_y, m_x, -c]

    #Center signum check
    pt_center_line = np.array([pt_center[0], pt_center[1], 1])  #Extend center point to be able to plug in line equation

    cent_sgn = np.dot(line_arr, pt_center_line)  #Plug center in line equation
    cent_sgn = np.sum(cent_sgn)                  #Sum equation to get value of line
    cent_sgn = np.sign(cent_sgn)                 #Get sign of center

    #Check point signum check
    check_pt_line = np.array([check_pt[0], check_pt[1], 1])  #Extend check point to be able to plug in line equation

    check_val = np.dot(line_arr, check_pt_line)    #Plug center in line equation
    check_val = np.sum(check_val)                  #Sum equation to get value of line
    check_sgn = np.sign(check_val)                 #Get sign of center

    if cent_sgn==check_sgn or check_sgn==0:
        
        return True

    div_val = np.sqrt(np.sum(np.square(line_arr[:2])))
    check_dist = np.abs(check_val)/div_val

    if check_dist<=10:
        
        return True
    

    radial_a = np.linalg.norm(np.subtract(check_pt, pt_a))
    radial_b = np.linalg.norm(np.subtract(check_pt, pt_b))

    if radial_a<=10 or radial_b<=10:
        
        return True
    
    return False

###########################################################################################################

def get_hex_pts():
    
    #Inner Hex
    hex_center = np.array([300,125])
    v_up = np.array([0,75])
    theta = np.deg2rad(60)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    hex_pts = []

    for i in range(6):
        v_up = np.dot(v_up, rot)
        new_pt = hex_center + v_up
        hex_pts.append(new_pt)

    hex_pts = np.array(hex_pts, np.int32)

    return hex_pts

###########################################################################################################

def half_plane_check(og_check_pt):

    hex_pts = get_hex_pts()
    check_pt = [og_check_pt[0],og_check_pt[1]]
    check = []

    #Rect 1 check
    check_1 = []
    check_1.append(line_check(np.array([100,0]), np.array([100,100]), np.array([125,50]), check_pt))
    check_1.append(line_check(np.array([100,100]), np.array([150,100]), np.array([125,50]), check_pt))
    check_1.append(line_check(np.array([150,100]), np.array([150,0]), np.array([125,50]), check_pt))
    check_1 = np.sum(check_1)
    check.append(check_1==3)

    #Rect 2 check
    check_2 = []
    check_2.append(line_check(np.array([100,250]), np.array([100,150]), np.array([125,200]), check_pt))
    check_2.append(line_check(np.array([100,150]), np.array([150,150]), np.array([125,200]), check_pt))
    check_2.append(line_check(np.array([150,150]), np.array([150,250]), np.array([125,200]), check_pt))
    check_2 = np.sum(check_2)
    check.append(check_2==3)

    #Hexagon check
    hex_center = np.array([300,125])
    hex_check = []
    hex_pts_here = hex_pts.tolist()

    for i in range(len(hex_pts_here)):

        hex_a = hex_pts_here[i]
        hex_b = hex_pts_here[i-1]

        hex_check.append(line_check(hex_a, hex_b, hex_center, check_pt))

    hex_check = np.sum(hex_check)
    check.append(hex_check>5)


    #Triangle check
    tri_pts = np.array([[460,25],[560,125],[460,225]])
    tri_center = np.average(tri_pts, axis=0)
    tri_check = []

    for i in range(len(tri_pts)):

        tri_a = tri_pts[i]
        tri_b = tri_pts[i-1]

        tri_check.append(line_check(tri_a, tri_b, tri_center, check_pt))

    tri_check = np.sum(tri_check)
    check.append(tri_check>2)

    check = np.sum(check)
    # print(check)
    return (check!=0)

#


###########################################################################################################

def gen_map(hex_pts):

    map_area = np.zeros((250,600,3), np.uint8)

    #Map area border
    cv2.rectangle(map_area, (0,0), (600,250), (255,0,0), -1)
    cv2.rectangle(map_area, (5,5), (595,245), (0,0,0),-1)

    #Rect 1
    cv2.rectangle(map_area, (100,100),(150,0),(255,0,0),5)
    cv2.rectangle(map_area, (100,100),(150,0),(0,255,0),-1)

    #Rect 2
    cv2.rectangle(map_area, (100,250),(150,150),(255,0,0),5)
    cv2.rectangle(map_area, (100,250),(150,150),(0,255,0),-1)

    #Inner Hex
    hex_center = np.array([300,125])

    #Outer Hex 
    hex_bdr = []
    for i in hex_pts:

        curr_vect = i - hex_center
        curr_vect_norm = np.linalg.norm(curr_vect)
        curr_vect = curr_vect + (curr_vect/curr_vect_norm)*5
        curr_vect = curr_vect + hex_center
        hex_bdr.append(curr_vect)

    hex_bdr = np.array(hex_bdr, np.int32)

    cv2.fillPoly(map_area, [hex_bdr], (255,0,0))
    cv2.fillPoly(map_area, [hex_pts], (0,255,0))

    #Triangle
    tri_pts = np.array([[460,25],[560,125],[460,225]])

    tri_center = np.average(tri_pts, axis=0)

    tri_bdr = []

    for i in tri_pts:

        curr_vect = i - tri_center
        curr_vect_norm = np.linalg.norm(curr_vect)
        curr_vect = curr_vect + (curr_vect/curr_vect_norm)*5
        curr_vect = curr_vect + tri_center
        tri_bdr.append(curr_vect)

    tri_bdr = np.array(tri_bdr, np.int32)
    cv2.fillPoly(map_area, [tri_bdr], (255,0,0))
    cv2.fillPoly(map_area, [tri_pts], (0,255,0))

    map_area = cv2.cvtColor(map_area,cv2.COLOR_BGR2RGB)
    return map_area


# Function to get optimal path
def get_pathpoints(check_key, initial):
    pathpoints.append(check_key)
    k = check_key
    while(k!=initial):
        
        k = closed_nodes[k][2]
        pathpoints.append(k)

def round_to_nearest_half(num):
    return round(num*2)/2

def Rotaion_matrix(l, phi, x,y, theta):
    theta = np.deg2rad(theta)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), x],[np.sin(theta), np.cos(theta), y ],[0 , 0, 1]])
    robot_movement = np.array([[l*np.cos(phi)],[l*np.sin(phi)],[1]])
    robot_coordinates = np.matmul(rot_mat,robot_movement)

    return robot_coordinates
    
# Defining action set
def action1(key,goal):
    l =step_size
    phi = 0
    new_coordinates = Rotaion_matrix(l,phi, key[0], key[1], key[2])
    new_coordinates[2] = key[2]
    new_coordinates = [np.rint(x) for x in new_coordinates]
    key_x = round_to_nearest_half(float(new_coordinates[0]))
    key_y = round_to_nearest_half(float(new_coordinates[1]))
    key_theta = round_to_nearest_half(float(new_coordinates[2]))%360
    child_key = (key_x,key_y,key_theta)
    create_child(key,child_key,goal)
    

def action2(key,goal):
    l =step_size
    phi = np.deg2rad(-30)
    new_coordinates = Rotaion_matrix(l,phi, key[0], key[1], key[2])
    new_coordinates[2] = key[2] + np.rad2deg(phi)
    new_coordinates = [np.rint(x) for x in new_coordinates]
    key_x = round_to_nearest_half(float(new_coordinates[0]))
    key_y = round_to_nearest_half(float(new_coordinates[1]))
    key_theta = round_to_nearest_half(float(new_coordinates[2]))%360
    child_key = (key_x,key_y,key_theta)
    create_child(key,child_key,goal)
    

def action3(key,goal):
    l =step_size
    phi = np.deg2rad(-60)
    new_coordinates = Rotaion_matrix(l,phi, key[0], key[1], key[2])
    new_coordinates[2] = key[2] + np.rad2deg(phi)
    new_coordinates = [np.rint(x) for x in new_coordinates]
    key_x = round_to_nearest_half(float(new_coordinates[0]))
    key_y = round_to_nearest_half(float(new_coordinates[1]))
    key_theta = round_to_nearest_half(float(new_coordinates[2]))%360
    child_key = (key_x,key_y,key_theta)
    create_child(key,child_key,goal)
    
    
def action4(key,goal):
    l =step_size
    phi = np.deg2rad(30)
    new_coordinates = Rotaion_matrix(l,phi, key[0], key[1], key[2])
    new_coordinates[2] = key[2] + np.rad2deg(phi)
    new_coordinates = [np.rint(x)for x in new_coordinates]
    key_x = round_to_nearest_half(float(new_coordinates[0]))
    key_y = round_to_nearest_half(float(new_coordinates[1]))
    key_theta = round_to_nearest_half(float(new_coordinates[2]))%360
    child_key = (key_x,key_y,key_theta)
    create_child(key,child_key,goal)
    

def action5(key,goal):
    l =step_size
    phi = np.deg2rad(60)
    new_coordinates = Rotaion_matrix(l,phi, key[0], key[1], key[2])
    new_coordinates[2] = key[2] + np.rad2deg(phi)
    new_coordinates = [np.rint(x) for x in new_coordinates]
    key_x = round_to_nearest_half(float(new_coordinates[0]))
    key_y = round_to_nearest_half(float(new_coordinates[1]))
    key_theta = round_to_nearest_half(float(new_coordinates[2]))%360
    child_key = (key_x,key_y,key_theta)
    create_child(key,child_key,goal)
    

def round_to_half(number):
    return (round(number*2))/2

def get_c2g(key,goal):
    return np.sqrt(((key[0]-goal[0])**2)+((key[1]-goal[1])**2))


def new_c2c(key1,key2):
    temp1 = key1[0] - key2[0]
    temp2 = key1[1] - key2[1]
    temp3 = temp1**2 + temp2**2

    return closed_nodes[key1][0] + np.sqrt(temp3)

def create_child(key,child1_key,goal):
    
    if if_node_goal(check_key,goal):
        goal_reached = 1
        get_pathpoints(check_key, initial)


    child_coordinates = [child1_key[0],child1_key[1]]
    if not child1_key in closed_nodes and not half_plane_check(child_coordinates) and 5 < child1_key[0] <595 and 5 < child1_key[1] <245 :
        
        c2c =  new_c2c(key,child1_key) 
        if not child1_key in open_nodes or open_nodes[child1_key][0] > 650:     
            
            total_cost = c2c+ get_c2g(child1_key,goal)
            open_nodes[child1_key] = [c2c,total_cost,key]
            explored_points.append(child1_key)
        else :
            if c2c < open_nodes[child1_key][0]:
                total_cost = c2c+ get_c2g(child1_key,goal)
                open_nodes[child1_key] = [c2c,total_cost,key]

    

# Function to check if node is goal coordinate
def if_node_goal(key,goal):
      
      if np.linalg.norm(np.array(key[:2])-np.array(goal[:2])) <= 1.5*5:
          return True
      else:
          return False


# main fucntion
open_nodes = {}
closed_nodes = {}
openList = []
closedList =[]
explored_points = []
obstacle_color = (0,0,255)
white_color = (255,255,255)
border_color = (150,150,255)
line_color = (255,0,0)
pathpoints = []
step_size  = 10 
threshold = 0.5
goal_reached = 0

# open_nodes = {}
wrong_coordinates = True
while wrong_coordinates:
    

    goal = (200,50)

    initial = (0,0)
    initial_list =[]
    initial_x = int(input("Enter start x coordinate "))
    initial_list.append(initial_x)
    initial_y = int(input("Enter start y coordinate "))
    initial_list.append(initial_y)
    initial_theta = int(input("Enter start theta "))
    initial_list.append(initial_theta)
    initial = tuple(initial_list)

    goal_list = []
    goal_x = int(input("Enter goal x coordinate "))
    goal_list.append(goal_x)
    goal_y = int(input("Enter goal y coordinate ")) 
    goal_list.append(goal_y)
    goal_theta = int(input("Enter start theta "))
    goal_list.append(goal_theta)
    goal = tuple(goal_list)

    
    # checking if given coordinates are not in obstacle
    if half_plane_check(initial):
        print("Start coordinates are in obstacle")
                    
    elif  half_plane_check(goal):
        print("Goal coordinates are in obstacle")
                    
    # checking if given coordinates are not in canvas
    elif initial_x < 0 or initial_x >= 600 or initial_y < 0 or initial_y >= 250:
        print("Start coordinates are out of canvas")
                    
    elif goal_x < 0 or goal_x >= 600 or  initial_y < 0 or initial_y >= 250:
        print("Goal coordinates are out of canvas")
                            
    else:
        wrong_coordinates = False

        
    check_flag = 0  

clearance = int(input('Enter clearance: '))
step_size = int(input('Enter step size: '))
open_nodes[initial] = [0,get_c2g(initial,goal),initial]

    
explored_points.append(initial)
goal_reached = if_node_goal(initial,goal)
i = 0

hex_pts = get_hex_pts()  
map_area = gen_map(hex_pts)        #Creates map area

while len(open_nodes) is not 0 and goal_reached is False:  

    i += 1
    open_nodes = dict(sorted(open_nodes.items(), key=lambda x:x[1][1],reverse = True))
    check_node = open_nodes.popitem()
    print("check node",check_node)
    check_key = check_node[0]
    closed_nodes[check_key] = [check_node[1][0],check_node[1][1],check_node[1][2]]
    curr_loc = [int(check_key[0]), int(check_key[1])]
    parent_loc = [int(check_node[1][2][0]), int(check_node[1][2][1])]

    map_area =cv2.arrowedLine(map_area, tuple(parent_loc), tuple(curr_loc), (245,66,206), 1)
    cv2.imshow('A Star', cv2.flip(map_area,0))
    if cv2.waitKey(1) & 0XFF == ord('q'):cv2.destroyAllWindows()


    if if_node_goal(check_key,goal):
        goal_reached = 1
        get_pathpoints(check_key, initial)
    else:
        action1(check_key, goal)
        action2(check_key, goal)
        action3(check_key, goal)
        action4(check_key, goal)
        action5(check_key, goal)
if goal_reached == 1:
    cv2.destroyAllWindows()
    print("Goal reached")

    path_new = []
    for i in pathpoints:

        path_new.append(i[:2])

    path_new = np.array(list(reversed(path_new)), np.int32)
    cv2.polylines(map_area, [path_new], False, (52,189,235), 1)

    for i in path_new:

        map_copy = map_area.copy()
        cv2.circle(map_copy, i, clearance, (180, 235, 52))
        cv2.imshow('A star backtrack', cv2.flip(map_copy,0))
        if cv2.waitKey(100) & 0xFF==ord('q'): cv2.destroyAllWindows()

    
