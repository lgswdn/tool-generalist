2-Finger Parallel Gripper. Each parallel gripper consists of 3 links, i.e., gripper base, left finger, right finger,
and 2 prismatic joints, i.e., the left finger joint and the right finger joint. The gripper base consists of a cube
and a cylinder with a random value to control the ratio. We also randomize the base geometry in terms of its
height, width, and depth. We create fingers as cubic objects and are modified based on the ratio of fingertip
width to the finger bottom width, and its tilting ratio. In half of all samples, we add additional meshes of
square, round, or triangle cylinders at the fingertip.
2-FingerRevoluteGripper. Each2-fingerrevolutegripperconsistsof5links, i.e., gripperbase, leftmidfinger,
left top finger, right mid finger, right top finger, and 4 revolute joints, i.e., left mid finger joint, left top finger
joins, right mid finger joint, and right top finger joint. The gripper base is randomized in dimension and in the
ratio between the base top and base bottom. The mid finger links and the top finger links are cubic objects.
For mid fingers, we randomly add an outer finger like in Robotiq-2F and OnRobot-RG grippers. For the top
fingers, we randomly add round / square cylinders at fingertips. There are two modes for gripper closing
motion. In the first mode, the right and left top finger links are always parallel to each other. Thus, the top
finger joint and the mid finger joint rotate in a ratio of 1 :−1. In the second mode, all top finger joints and
the mid finger joints rotate in a ratio of 1 : 1, where the gripper fingers will close like a pinch gripper.
3-Finger High-DOF Grippers. Each 3-finger high-dof gripper consists of the gripper base and 3 2-joint / 3-
joint fingers. We create a cubic gripper base with randomized dimensions and wrist-to-palm ratio. We attach
two fingers at the top of the palm and one finger at the center of the wrist. All fingers are stretching in x-axis
when it is open. The total DOFs with 2-joint fingers is 6 and the total DOF with 3-joint fingers is 9. All finger
links are cubic objects with random width, depth, and height. Moreover, we randomly change the orientation
ofthetwofingersatthetopofthepalmaroundthez-axistomimicthepotentialsiderotationthatreal3-finger
handshave(e.g., Robotiq-3F).Forthegripperclosingmotion, alljointsfollowalinearlyinterpolatedtrajectory
from the fully open state to the fully closed state.