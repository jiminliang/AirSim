import numpy
import cv2
import time
import sys
import os
import random
import glob
from airsim import *
from create_ir_segmentation_map import get_new_temp_emiss_from_radiance, set_segmentation_ids

def rotation_matrix_from_angles(pry):
    pitch = pry[0]
    roll = pry[1]
    yaw = pry[2]
    sy = numpy.sin(yaw)
    cy = numpy.cos(yaw)
    sp = numpy.sin(pitch)
    cp = numpy.cos(pitch)
    sr = numpy.sin(roll)
    cr = numpy.cos(roll)
    
    Rx = numpy.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    
    Ry = numpy.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    
    Rz = numpy.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])
    
    #Roll is applied first, then pitch, then yaw.
    RyRx = numpy.matmul(Ry, Rx)
    return numpy.matmul(Rz, RyRx)

def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    #Turn the camera position into a column vector.
    camPosition = numpy.transpose([camXYZ])

    #Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = utils.to_eularian_angles(camQuaternion)
    
    #Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = rotation_matrix_from_angles(pitchRollYaw)
    
    #Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = numpy.transpose([subjectXYZ])
    XYZW = numpy.add(XYZW, -camPosition)
    print("XYZW: " + str(XYZW))
    XYZW = numpy.matmul(numpy.transpose(camRotation), XYZW)
    print("XYZW derot: " + str(XYZW))
    
    #Recreate the perspective projection of the camera.
    XYZW = numpy.concatenate([XYZW, [[1]]])    
    XYZW = numpy.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]
    
    #Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2
    
    return numpy.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY
    ]).reshape(2,)
   
def get_image(x, y, z, pitch, roll, yaw, client):
    """
    title::
        get_image

    description::
        Capture images (as numpy arrays) from a certain position.

    inputs::
        x
            x position in meters
        y
            y position in meters
        z
            altitude in meters; remember NED, so should be negative to be 
            above ground
        pitch
            angle (in radians); in computer vision mode, this is camera angle
        roll
            angle (in radians)
        yaw
            angle (in radians)
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)

    returns::
        position
            AirSim position vector (access values with x_val, y_val, z_val)
        angle
            AirSim quaternion ("angles")
        im
            segmentation or IR image, depending upon palette in use (3 bands)
        imScene
            scene image (3 bands)

    author::
        Elizabeth Bondi
        Shital Shah
    """

    #Set pose and sleep after to ensure the pose sticks before capturing image.
    client.simSetVehiclePose(Pose(Vector3r(x, y, z), \
                      to_quaternion(pitch, roll, yaw)), True)
    time.sleep(0.1)

    #Capture segmentation (IR) and scene images.
    responses = \
        client.simGetImages([ImageRequest("0", ImageType.Infrared,
                                          False, False),
                            ImageRequest("0", ImageType.Scene, \
                                          False, False),
                            ImageRequest("0", ImageType.Segmentation, \
                                          False, False)])

    #Change images into numpy arrays.
    img1d = numpy.fromstring(responses[0].image_data_uint8, dtype=numpy.uint8)
    im = img1d.reshape(responses[0].height, responses[0].width, 3) 

    img1dscene = numpy.fromstring(responses[1].image_data_uint8, dtype=numpy.uint8)
    imScene = img1dscene.reshape(responses[1].height, responses[1].width, 3)

    img1dseg = numpy.fromstring(responses[2].image_data_uint8, dtype=numpy.uint8)
    imSeg = img1dseg.reshape(responses[2].height, responses[2].width, 3)

    return Vector3r(x, y, z), to_quaternion(pitch, roll, yaw), im, imScene, imSeg

def main(client,
         objectList,
         pitch=numpy.radians(270), #image straight down
         roll=0,
         yaw=0,
         z=-122,
         writeIR=True,
         writeScene=True,
         writeSeg=True,
         irFolder='',
         sceneFolder='',
         segFolder=''):
    """
    title::
        main

    description::
        Follow objects of interest and record images while following.

    inputs::
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)
        objectList
            list of tag names within the AirSim environment, corresponding to 
            objects to follow (add tags by clicking on object, going to 
            Details, Actor, and Tags, then add component)
        pitch
            angle (in radians); in computer vision mode, this is camera angle
        roll
            angle (in radians)
        yaw
            angle (in radians)
        z
            altitude in meters; remember NED, so should be negative to be 
            above ground
        write
            if True, will write out the images
        folder
            path to a particular folder that should be used (then within that
            folder, expected folders are ir and scene)

    author::
        Elizabeth Bondi
    """
    i = 0
    for o in objectList:
        startTime = time.time()
        elapsedTime = 0
        pose = client.simGetObjectPose(o)

        #Capture images for a certain amount of time in seconds (half hour now)
        # while elapsedTime < 5:
        for i in range(1):    # capture fixed number of images, by Jimin
            #Capture image - pose.position x_val access may change w/ AirSim
            #version (pose.position.x_val new, pose.position[b'x_val'] old)
            vector, angle, ir, scene, seg = get_image(pose.position.x_val, 
                                                    pose.position.y_val, 
                                                    z, 
                                                    pitch, 
                                                    roll, 
                                                    yaw, 
                                                    client)

            #Convert color scene image to BGR for write out with cv2.
            # r,g,b = cv2.split(scene)
            # scene = cv2.merge((b,g,r))

            if writeIR:
                cv2.imwrite(os.path.join(irFolder,    o+'_'+str(i).zfill(5)+'_ir'+'.png'), ir)
            if writeScene:
                cv2.imwrite(os.path.join(sceneFolder, o+'_'+str(i).zfill(5)+'_rgb'+'.png'), scene)
            if writeSeg:
                cv2.imwrite(os.path.join(sceneFolder, o+'_'+str(i).zfill(5)+'_seg'+'.png'), seg)

            i += 1
            elapsedTime = time.time() - startTime
            pose = client.simGetObjectPose(o)
            camInfo = client.simGetCameraInfo("0")
            object_xy_in_pic = project_3d_point_to_screen(
                [pose.position.x_val, pose.position.y_val, pose.position.z_val],
                [camInfo.pose.position.x_val, camInfo.pose.position.y_val, camInfo.pose.position.z_val],
                camInfo.pose.orientation,
                camInfo.proj_mat.matrix,
                ir.shape[:2][::-1]
            )
            print("Object projected to pixel\n{!s}.".format(object_xy_in_pic))

def set_IR_parameters(client, season='winter'):
    segIdDict = {'Base_Terrain':'soil',
                 'elephant':'elephant',
                 'zebra':'zebra',
                 'Crocodile':'crocodile',
                 'Rhinoceros':'rhinoceros',
                 'Hippo':'hippopotamus',
                 'Poacher':'human',
                 'InstancedFoliageActor':'tree',
                 'Water_Plane':'water',
                 'truck':'truck'}
    
    #Choose temperature values for winter or summer.
    #"""
    #winter
    if season == 'winter':
        tempEmissivity = numpy.array([['elephant',290,0.96], 
                                  ['zebra',298,0.98],
                                  ['rhinoceros',291,0.96],
                                  ['hippopotamus',290,0.96],
                                  ['crocodile',295,0.96],
                                  ['human',292,0.985], 
                                  ['tree',273,0.952], 
                                  ['grass',273,0.958], 
                                  ['soil',278,0.914], 
                                  ['shrub',273,0.986],
                                  ['truck',273,0.8],
                                  ['water',273,0.96]])

    #summer
    if season == 'summer':
        tempEmissivity = numpy.array([['elephant',298,0.96], 
                                  ['zebra',307,0.98],
                                  ['rhinoceros',299,0.96],
                                  ['hippopotamus',298,0.96],
                                  ['crocodile',303,0.96],
                                  ['human',301,0.985], 
                                  ['tree',293,0.952], 
                                  ['grass',293,0.958], 
                                  ['soil',288,0.914], 
                                  ['shrub',293,0.986],
                                  ['truck',293,0.8],
                                  ['water',293,0.96]])

    #Read camera response.
    response = None
    camResponseFile = 'camera_response.npy'
    try:
      numpy.load(camResponseFile)
    except:
      print("{} not found. Using default response.".format(camResponseFile))

    #Calculate radiance.
    tempEmissivityNew = get_new_temp_emiss_from_radiance(tempEmissivity, 
                                                         response)

    #Set IDs in AirSim environment.
    set_segmentation_ids(segIdDict, tempEmissivityNew, client)

if __name__ == '__main__':
    
    #Connect to AirSim, UAV mode.
    client = MultirotorClient()
    client.confirmConnection()

    set_IR_parameters(client=client, season='winter')

    allObjects = client.simListSceneObjects('.*')

    #Look for objects with names that match a regular expression.
    poacherList = client.simListSceneObjects('.*?Poacher.*?')
    elephantList = client.simListSceneObjects('.*?Elephant.*?')
    crocList = client.simListSceneObjects('.*?Croc.*?')
    hippoList = client.simListSceneObjects('.*?Hippo.*?')
    
    objectList = poacherList + elephantList + crocList + hippoList
    
    #Sample calls to main, varying camera angle and altitude.
    #straight down, 400ft
    dataFolder=r'..\Sim_data\winter\400ft\down'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    main(client, 
         objectList, 
         irFolder=dataFolder,
         sceneFolder=dataFolder,
         segFolder=dataFolder) 
    
    #45 degrees, 400ft -- note that often object won't be scene since position
    #is set exactly to object's
    dataFolder=r'..\Sim_data\winter\400ft\45'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    main(client, 
         objectList, 
         pitch=numpy.radians(315), 
         irFolder=dataFolder,
         sceneFolder=dataFolder,
         segFolder=dataFolder)  

    #straight down, 200ft
    dataFolder=r'..\Sim_data\winter\200ft\down'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    main(client, 
         objectList, 
         z=-61, 
         irFolder=dataFolder,
         sceneFolder=dataFolder,
         segFolder=dataFolder)
    
    #45 degrees, 200ft -- note that often object won't be scene since position
    #is set exactly to object's
    dataFolder=r'..\Sim_data\winter\200ft\45'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    main(client, 
         objectList, 
         z=-61, 
         pitch=numpy.radians(315), 
         irFolder=dataFolder,
         sceneFolder=dataFolder,
         segFolder=dataFolder)