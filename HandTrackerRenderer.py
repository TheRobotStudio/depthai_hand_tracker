import time
from copy import deepcopy

import cv2
import numpy as np
import math
import struct
import serial

from mediapipe_utils import angle

#from Angle import HandAngles

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port="COM4",
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    xonxoff=False,
    rtscts=False,
    dsrdtr=False,
    timeout=1
)

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

# LINES_BODY to draw the body skeleton when Body Pre Focusing is used
LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]

class HandTrackerRenderer:
    def __init__(self, 
                tracker,
                output=None):

        self.tracker = tracker

        # Rendering flags
        if self.tracker.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = 0
            self.show_landmarks = True
            self.show_scores = False
            self.show_gesture = self.tracker.use_gesture
        else:
            self.show_pd_box = True
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_scores = False

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz
        self.show_fps = True
        self.show_body = False # self.tracker.body_pre_focusing is not None
        self.show_inferences_status = False

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h)) 

    def norm2abs(self, x_y):
        x = int(x_y[0] * self.tracker.frame_size - self.tracker.pad_w)
        y = int(x_y[1] * self.tracker.frame_size - self.tracker.pad_h)
        return (x, y)

    def draw_hand(self, hand):

        # Send hand info
        #print(hand.landmarks[0, 0])
        #print(hand.landmarks[0, 1])
        #print(hand.norm_landmarks[1, 1], hand.norm_landmarks[1, 1], hand.norm_landmarks[1, 2])
        #print(hand.world_landmarks[1,2])
        # choose packing number of and size, e.g. 2h means 2 short integers (2 bytes), f means float
        #packed_data = struct.pack('2h', hand.landmarks[0, 0], hand.landmarks[0, 1])

        #packed_data = struct.pack('h', hand.landmarks[0, 0])

        #sending integers works fine to the Arudino
        # landmarks are 2D
        # norm_landmarks are 3D but floats so need to do the angle calcs here and send results to the servos as commands
        #packed_data = struct.pack('2f', hand.norm_landmarks[1, 0], hand.norm_landmarks[1, 1])

        #ser.write(packed_data)

        #ser.write(b"\n")

        #angle = HandAngles.fingerAngles(hand.landmarks)

        #finger = angle(hand.norm_landmarks[1, 0], hand.norm_landmarks[1, 1], hand.norm_landmarks[1, 2])
        #a_joint = np.array([hand.norm_landmarks[7, 0], hand.norm_landmarks[7, 1], hand.norm_landmarks[7, 2]])
        #b_joint = np.array([hand.norm_landmarks[6, 0], hand.norm_landmarks[6, 1], hand.norm_landmarks[6, 2]])
        #c_joint = np.array([hand.norm_landmarks[5, 0], hand.norm_landmarks[5, 1], hand.norm_landmarks[5, 2]])
        #finger = angle(a_joint, b_joint, c_joint)
        #print(finger)

        joint_xyz = np.zeros((21, 3))
        for i in range(21):
            joint_xyz[i] = ([hand.norm_landmarks[i, 0], hand.norm_landmarks[i, 1], hand.norm_landmarks[i, 2]])
        # Create array with enough space for all calculated angles
        joint_angles = np.zeros(23)

        # First finger, fore or index
        # Angles calculated correspond to knuckle flex, knuckle yaw and long tendon length for all fingers,
        # note difference in knuckle yaw for little
        joint_angles[0] = angle(joint_xyz[0], joint_xyz[5], joint_xyz[8])
        joint_angles[1] = angle(joint_xyz[9], joint_xyz[5], joint_xyz[6])
        joint_angles[2] = angle(joint_xyz[5], joint_xyz[6], joint_xyz[7])

        # Second finger, middle
        joint_angles[3] = angle(joint_xyz[0], joint_xyz[9], joint_xyz[12])
        joint_angles[4] = angle(joint_xyz[13], joint_xyz[9], joint_xyz[10])
        joint_angles[5] = angle(joint_xyz[9], joint_xyz[10], joint_xyz[11])

        # Third finger, ring
        joint_angles[6] = angle(joint_xyz[0], joint_xyz[13], joint_xyz[16])
        joint_angles[7] = angle(joint_xyz[9], joint_xyz[13], joint_xyz[14])
        joint_angles[8] = angle(joint_xyz[13], joint_xyz[14], joint_xyz[15])

        # Fourth finger, pinky
        joint_angles[9] = angle(joint_xyz[0], joint_xyz[17], joint_xyz[20])
        joint_angles[10] = angle(joint_xyz[13], joint_xyz[17], joint_xyz[18])
        joint_angles[11] = angle(joint_xyz[17], joint_xyz[18], joint_xyz[19])

        # Thumb, bit of a guess for basal rotation might be better automatic
        joint_angles[12] = angle(joint_xyz[1], joint_xyz[2], joint_xyz[3])
        joint_angles[13] = angle(joint_xyz[2], joint_xyz[1], joint_xyz[5])
        joint_angles[14] = angle(joint_xyz[2], joint_xyz[3], joint_xyz[4])
        joint_angles[15] = angle(joint_xyz[9], joint_xyz[5], joint_xyz[2])
        #down = [x[:] for x in joint_xyz[0]]
        down = deepcopy(joint_xyz[0])
        #print(down, joint_xyz[0])
        down[2] = down[2] - 0.1
        #print(down, joint_xyz[0])
        joint_angles[16] = angle(joint_xyz[9], joint_xyz[0], down)
        joint_angles[16] = np.clip(joint_angles[16], 0, 180)
        #print(joint_angles[16])

        joint_angles[17] = math.degrees(hand.rotation) + 45
        #print(joint_angles[17])
        joint_angles[17] = np.clip(joint_angles[17], 0, 90)

        towards = deepcopy(joint_xyz[17])
        towards[2] = towards[2] - 0.1
        joint_angles[18] = angle(joint_xyz[5], joint_xyz[17], towards)
        joint_angles[18] = np.clip(joint_angles[18], 0, 180)
        print(joint_angles[16], joint_angles[17], joint_angles[18])

        # Prvents serial empty errors during development
        #for x in range(19, 23):
        #    joint_angles[x] = 123

        # For shoulder there is only the xyz of the wrist for now, need the holistic mediapipe model.
        # Shoulder pitch
        joint_angles[19] = 123

        # Shoulder Yaw
        joint_angles[20] = 123

        # Shoulder Roll
        joint_angles[21] = 123

        # ELbow
        joint_angles[22] = 123

        # Convert to int before sending over serial, increase value first to offset lost resolution
        #joint_angles = joint_angles * 10
        joint_angles = joint_angles.astype(int)
        #joint_angles = joint_angles.astype(bytes)
        print(joint_angles)

        #print(hand.world_landmarks[9, 1], hand.world_landmarks[0, 1],
        # hand.world_landmarks[9, 2], hand.world_landmarks[0, 2])
        #delta_y = hand.world_landmarks[9, 1] - hand.world_landmarks[0, 1]
        #delta_z = hand.world_landmarks[9, 2] - hand.world_landmarks[0, 2]
        #print(math.degrees(math.atan(delta_y / delta_z)))
        #print(math.degrees(hand.rotation))
        #print(math.degrees(math.atan(1)))


        # Generate checksum
        sum = np.sum(joint_angles)
        sum = sum & 0x000000FF
        t_xchecksum = 255 - sum
        #print(sum, t_xchecksum)

        #t_xchecksum = chr(t_xchecksum)
        #t_xchecksum = t_xchecksum & 0x000000FF
        #print(sum, t_xchecksum)
        # choose packing number of and size, e.g. 2h means 2 short integers (2 bytes), f means float
        #packed_data = struct.pack('2B', joint_angles[0], joint_angles[1])

        #delay = 0.0001

        # Initialise bus with two A
        command = B'\xFE'
        ser.write(command)
        ser.write(command)
        packed_data = struct.pack('23B', *joint_angles)
        ser.write(packed_data)
        #packed_data = struct.pack('B', t_xchecksum)
        #ser.write(packed_data)
        #ser.flushOutput()

        #print(ser.out_waiting)

        #time.sleep(delay)



        #packed_data = struct.pack('2B', joint_angles[0], joint_angles[1])
        #ser.write(packed_data)
        #ser.flushOutput()

        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[2], joint_angles[3])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[4], joint_angles[5])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[6], joint_angles[7])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[8], joint_angles[9])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[10], joint_angles[11])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[12], joint_angles[13])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)

        #packed_data = struct.pack('2B', joint_angles[14], joint_angles[15])
        #ser.write(packed_data)
        #ser.flushOutput()
        #time.sleep(delay)



        # End bus with check sum and two B
        ser.write(struct.pack('B', t_xchecksum))
        command = B'\xFD'
        ser.write(command)
        ser.write(command)

        #time.sleep(delay)

        ser.flushOutput()

        if self.tracker.use_lm:
            # (info_ref_x, info_ref_y): coords in the image of a reference point 
            # relatively to which hands information (score, handedness, xyz,...) are drawn
            info_ref_x = hand.landmarks[0,0]
            info_ref_y = np.max(hand.landmarks[:,1])

            # thick_coef is used to adapt the size of the draw landmarks features according to the size of the hand.
            thick_coef = hand.rect_w_a / 400
            if hand.lm_score > self.tracker.lm_score_thresh:
                if self.show_rot_rect:
                    cv2.polylines(self.frame, [np.array(hand.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
                if self.show_landmarks:
                    lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int) for line in LINES_HAND]
                    if self.show_handedness == 3:
                        color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                    else:
                        color = (255, 0, 0)
                    cv2.polylines(self.frame, lines, False, color, int(1+thick_coef*3), cv2.LINE_AA)
                    radius = int(1+thick_coef*5)
                    if self.tracker.use_gesture:
                        # color depending on finger state (1=open, 0=close, -1=unknown)
                        color = { 1: (0,255,0), 0: (0,0,255), -1:(0,255,255)}
                        cv2.circle(self.frame, (hand.landmarks[0][0], hand.landmarks[0][1]), radius, color[-1], -1)
                        for i in range(1,5):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.thumb_state], -1)
                        for i in range(5,9):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.index_state], -1)
                        for i in range(9,13):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.middle_state], -1)
                        for i in range(13,17):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.ring_state], -1)
                        for i in range(17,21):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.little_state], -1)
                    else:
                        if self.show_handedness == 2:
                            color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                        elif self.show_handedness == 3:
                            color = (255, 0, 0)
                        else: 
                            color = (0,128,255)
                        for x,y in hand.landmarks[:,:2]:
                            cv2.circle(self.frame, (int(x), int(y)), radius, color, -1)

                if self.show_handedness == 1:
                    cv2.putText(self.frame, f"{hand.label.upper()} {hand.handedness:.2f}", 
                            (info_ref_x-90, info_ref_y+40), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if hand.handedness > 0.5 else (0,0,255), 2)
                if self.show_scores:
                    cv2.putText(self.frame, f"Landmark score: {hand.lm_score:.2f}", 
                            (info_ref_x-90, info_ref_y+110), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if self.tracker.use_gesture and self.show_gesture:
                    cv2.putText(self.frame, hand.gesture, (info_ref_x-20, info_ref_y-50), 
                            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        if hand.pd_box is not None:
            box = hand.pd_box
            box_tl = self.norm2abs((box[0], box[1]))
            box_br = self.norm2abs((box[0]+box[2], box[1]+box[3]))
            if self.show_pd_box:
                cv2.rectangle(self.frame, box_tl, box_br, (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(hand.pd_kps):
                    x_y = self.norm2abs(kp)
                    cv2.circle(self.frame, x_y, 6, (0,0,255), -1)
                    cv2.putText(self.frame, str(i), (x_y[0], x_y[1]+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                if self.tracker.use_lm:
                    x, y = info_ref_x - 90, info_ref_y + 80
                else:
                    x, y = box_tl[0], box_br[1]+60
                cv2.putText(self.frame, f"Palm score: {hand.pd_score:.2f}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            
        if self.show_xyz:
            if self.tracker.use_lm:
                x0, y0 = info_ref_x - 40, info_ref_y + 40
            else:
                x0, y0 = box_tl[0], box_br[1]+20
            cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
            cv2.putText(self.frame, f"X:{hand.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
            cv2.putText(self.frame, f"Y:{hand.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            cv2.putText(self.frame, f"Z:{hand.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        if self.show_xyz_zone:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(hand.xyz_zone[0:2]), tuple(hand.xyz_zone[2:4]), (180,0,180), 2)

    def draw_body(self, body):
        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.tracker.body_score_thresh and body.scores[line[1]] > self.tracker.body_score_thresh]
        cv2.polylines(self.frame, lines, False, (255, 144, 30), 2, cv2.LINE_AA)

    def draw_bag(self, bag):
        
        if self.show_inferences_status:
            # Draw inferences status
            h = self.frame.shape[0]
            u = h // 10
            status=""
            if bag.get("bpf_inference", 0):
                cv2.rectangle(self.frame, (u, 8*u), (2*u, 9*u), (255,144,30), -1)
            if bag.get("pd_inference", 0):
                cv2.rectangle(self.frame, (2*u, 8*u), (3*u, 9*u), (0,255,0), -1)
            nb_lm_inferences = bag.get("lm_inference", 0)
            if nb_lm_inferences:
                cv2.rectangle(self.frame, (3*u, 8*u), ((3+nb_lm_inferences)*u, 9*u), (0,0,255), -1)

        body = bag.get("body", False)
        if body and self.show_body:
            # Draw skeleton
            self.draw_body(body)
            # Draw Movenet smart cropping rectangle
            cv2.rectangle(self.frame, (body.crop_region.xmin, body.crop_region.ymin), (body.crop_region.xmax, body.crop_region.ymax), (0,255,255), 2)
            # Draw focus zone
            focus_zone= bag.get("focus_zone", None)
            if focus_zone:
                cv2.rectangle(self.frame, tuple(focus_zone[0:2]), tuple(focus_zone[2:4]), (0,255,0),2)

    def draw(self, frame, hands, bag={}):
        self.frame = frame
        if bag:
            self.draw_bag(bag)
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        if self.show_fps:
                self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("Snapshot saved in snapshot.jpg")
                cv2.imwrite("snapshot.jpg", self.frame)
        elif key == ord('1'):
            self.show_pd_box = not self.show_pd_box
        elif key == ord('2'):
            self.show_pd_kps = not self.show_pd_kps
        elif key == ord('3'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('4') and self.tracker.use_lm:
            self.show_landmarks = not self.show_landmarks
        elif key == ord('5') and self.tracker.use_lm:
            self.show_handedness = (self.show_handedness + 1) % 4
        elif key == ord('6'):
            self.show_scores = not self.show_scores
        elif key == ord('7') and self.tracker.use_lm:
            if self.tracker.use_gesture:
                self.show_gesture = not self.show_gesture
        elif key == ord('8'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('9'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('b'):
            try:
                if self.tracker.body_pre_focusing:
                    self.show_body = not self.show_body 
            except:
                pass
        elif key == ord('s'):
            self.show_inferences_status = not self.show_inferences_status
        return key
