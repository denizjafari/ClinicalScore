import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import VideoWriter, VideoWriter_fourcc
import pandas as pd
import torch
from scipy import signal, ndimage, spatial
import math


def _gaussian_fast(size=3, sigma=0.25, amplitude=1., offset=[0., 0.], device='cpu'):
    coordinates = torch.stack(torch.meshgrid(torch.arange(-size // 2 + 1. -offset[0], size // 2 + 1. -offset[0], step=1),
                                   torch.arange(-size // 2 + 1. -offset[1], size // 2 + 1. -offset[1], step=1))).to(device)
    coordinates = coordinates / (sigma * size)
    gauss = amplitude * torch.exp(-(coordinates**2 / 2).sum(dim=0))
    return gauss.permute(1, 0)


def draw_gaussian(image, point, sigma, offset=False):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    # g = torch.Tensor(_gaussian(size, offset=point%1) if offset else _gaussian(size), device=image.device)
    g = _gaussian_fast(size, offset=point%1, device=image.device) if offset else _gaussian_fast(size, device=image.device)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

def transform(point, center, scale, resolution, invert=False, integer=True):
    """Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int() if integer else new_point

def create_target_heatmap(target_landmarks, centers, scales, size = 64):
    """
    Receives a batch of landmarks and returns a set of heatmaps for each set of 68 landmarks in the batch
    :param target_landmarks: the batch is expected to have the dim (n x 68 x 2). Where n is the batch size
    :return: returns a (n x 68 x 64 x 64) batch of heatmaps
    """
    heatmaps = np.zeros((target_landmarks.shape[0], 68, size, size), dtype=np.float32)
    for i in range(heatmaps.shape[0]):
        for p in range(68):
            # Lua code from https://github.com/1adrianb/face-alignment-training/blob/master/dataset-images.lua:
            # drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, 64), 1)
            # Not sure why it adds 1 to each landmark before transform.
            landmark_cropped_coor = transform(target_landmarks[i, p] + 1, centers[i], scales[i], 64, invert=False)
            heatmaps[i, p] = draw_gaussian(heatmaps[i, p], landmark_cropped_coor + 1, 1)
    return torch.tensor(heatmaps)

def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps
    Note: Tried moving this to GPU, but realized it doesn't make sense.
    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(0, -ul[0]), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(0, -ul[1]), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(0, ul[0]), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(0, ul[1]), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0]:newY[1], newX[0]:newX[1]] = image[oldY[0]:oldY[1], oldX[0]:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg


def get_preds_fromhm_subpixel(hm, center=None, scale=None):
    """Similar to `get_preds_fromhm` Except it tries to estimate the coordinates of the mode of the distribution.
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0] = (preds[..., 0]) % hm.size(3)
    preds[..., 1].div_(hm.size(2)).floor_()
    eps = torch.tensor(0.0000000001).to(hm.device)
    # This is a magic number as far as understand.
    # 0.545 reduces the quantization error to exactly zero when `scale` is ~1.
    # 0.555 reduces the quantization error to exactly zero when `scale` is ~3.
    # 0.560 reduces the quantization error to exactly zero when `scale` is ~4.
    # 0.565 reduces the quantization error to exactly zero when `scale` is ~5.
    # 0.580 reduces the quantization error to exactly zero when `scale` is ~10.
    # 0.5825 reduces the quantization error to <0.002RMSE  when `scale` is ~100.
    sigma = 0.55
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            x0 = pX
            y0 = pY
            p0 = torch.max(hm_[pY, pX], eps)
            if pX < 63:
                p1 = torch.max(hm_[pY, pX + 1], eps)
                x1 = x0 + 1
                y1 = y0
                x = (3 * sigma)**2 * (torch.log(p1) - torch.log(p0)) - (
                            x0**2 - x1**2 + y0**2 - y1**2) / 2
                if pY < 63:
                    p2 = torch.max(hm_[pY + 1, pX], eps)
                    x2 = x0
                    y2 = y0 + 1
                    y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                x0**2 - x2**2 + y0**2 - y2**2) / 2
                else:
                    p2 = torch.max(hm_[pY - 1, pX], eps)
                    x2 = x0
                    y2 = y1 - 1
                    y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                x0**2 - x2**2 + y0**2 - y2**2) / 2
            else:
                p1 = torch.max(hm_[pY, pX - 1], eps)
                x1 = x0 - 1
                y1 = y0
                x = (3 * sigma)**2 * (torch.log(p1) - torch.log(p0)) - (
                            x0**2 - x1**2 + y0**2 - y1**2) / 2
                if pY < 63:
                    p2 = torch.max(hm_[pY + 1, pX], eps)
                    x2 = x0
                    y2 = y0 + 1
                    y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                x0**2 - x2**2 + y0**2 - y2**2) / 2
                else:
                    p2 = torch.max(hm_[pY - 1, pX])
                    x2 = x0
                    y2 = y1 - 1
                    y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                x0**2 - x2**2 + y0**2 - y2**2) / 2
            preds[i, j, 0] = x
            preds[i, j, 1] = y
    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j]+0.5, center[i], scale[i], hm.size(2), True)
    return preds, preds_orig


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0] = (preds[..., 0]) % hm.size(3)
    preds[..., 1].div_(hm.size(2)).floor_()

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]]).to(preds.device)
                preds[i, j].add_(diff.sign_().mul_(.25))
    preds.add_(+.5)
    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(preds[i, j], center[i], scale[i], hm.size(2), True)
    return preds, preds_orig

def get_color_video(BAG_File):

    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, BAG_File, repeat_playback=False)

    config.enable_all_streams()
    profile = pipeline.start(config)

    # create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # inform the device that this is not live streaming from camera
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    duration = playback.get_duration()

    true_frame_number = []
    frame_number = []
    time_st = []

    num_frame = 0


    Color_Frames = []#{}

    try:
        while True:
            frames = pipeline.wait_for_frames()  #get frame from file

            this_frame = frames.get_frame_number()  #get frame number 

            if (num_frame != 0) and (true_frame_number[-1] == this_frame): #verify that frame number is not repeated 
                #if frame number is repeated then replace the stored information 
                aligned_frames = align.process(frames)

                #take color and depth from frame, if any to these is not available then skip the frame
                aligned_depth = aligned_frames.get_depth_frame()
                aligned_color = aligned_frames.get_color_frame()

                # validate that both frames are available
                if not aligned_depth or not aligned_color:
                    continue

                time_stamp = frames.get_timestamp()
                true_frame_number[-1] = frames.get_frame_number()
                time_st[-1] = time_stamp 

                # transform to np array

                color_data = np.asanyarray(aligned_color.as_frame().get_data(), dtype=np.int)
                #depth_data = np.asanyarray(aligned_depth.as_frame().get_data(), dtype=np.int)
                # adjust depth data in meters
                #depth_data *= depth_scale

                Color_Frames[-1] = color_data

            else:
                #if frame number is not repeated then append the stored information 
                aligned_frames = align.process(frames)

                #take color and depth from frame, if any to these is not available then skip the frame
                aligned_depth = aligned_frames.get_depth_frame()
                aligned_color = aligned_frames.get_color_frame()

                # validate that both frames are available
                if not aligned_depth or not aligned_color:
                    continue

                time_stamp = frames.get_timestamp()
                true_frame_number.append(frames.get_frame_number())
                time_st.append(time_stamp )

                # transform to np array

                color_data = np.asanyarray(aligned_color.as_frame().get_data(), dtype=np.int)
                #depth_data = np.asanyarray(aligned_depth.as_frame().get_data(), dtype=np.int)
                # adjust depth data in meters
                #depth_data *= depth_scale

                Color_Frames.append(color_data)
                #Depth_Frames.append(depth_data

                frame_number.append(num_frame)
                num_frame += 1

    except RuntimeError:
        pass
    finally:
        pipeline.stop()
        
    duration_movie = duration.total_seconds()
    FPS = num_frame/duration_movie
    height, width,_ =  Color_Frames[0].shape

#     color_file = BAG_File[:-4]+'_color.mp4'

#     video = VideoWriter(color_file, 0x00000021, int(FPS), (width,height))

    color_file = BAG_File[:-4]+'_color.avi'

    video = VideoWriter(color_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(FPS), (width,height))

    for k in range(num_frame):
        frame_to_save = Color_Frames[k].astype('uint8')
        video.write(frame_to_save)

    video.release()    


    cvs_frame_info = BAG_File[:-4]+'_frameInfoColor.csv'
    df_cols = ['BAG_Frame_Number', 'Frame_Time_Stamp', 'Frame_Number_in_Video']
    df = pd.DataFrame(columns=df_cols)
    df['BAG_Frame_Number'] = true_frame_number
    df['Frame_Time_Stamp'] = (np.array(time_st)-time_st[0])/1000
    df['Frame_Number_in_Video'] = frame_number

    df.to_csv(cvs_frame_info)
    #print('success reading BAG file')
    
    return color_file, cvs_frame_info


# ### This portion of the notebook takes the color information and finds the position of facial landmarks using FAN

# #### Load models for face and facial landmarks localization

# In[34]:


def find_landmarks(BAG_File,
                   device,
                   color_file,
                   cvs_frame_info,
                   localize_face=0,
                   sufix = 'landmarks',
                   batch_size = 10,
                   fix_head_position=True,
                   face_alignment_net=None,
                   face_detection_net=None):
    
    import face_alignment.utils as utils
    
    if (face_alignment_net is None) or (face_detection_net is None):       
        from face_alignment import api as face_alignment
        from face_alignment.models import FAN
        from face_alignment.detection.sfd import sfd_detector

        def load_weights(model, filename):
            sd = torch.load(filename, map_location=lambda storage, loc: storage)
            names = set(model.state_dict().keys())
            for n in list(sd.keys()): 
                if n not in names and n+'_raw' in names:
                    if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
                    del sd[n]
            model.load_state_dict(sd)
            
        #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if face_alignment_net is None:
            face_alignment_model = r"./models/2DFAN4-11f355bf06.pth.tar"    
            #Face alignement
            network_size = 4
            face_alignment_net = FAN(network_size)
            load_weights(face_alignment_net,face_alignment_model)
            face_alignment_net.to(device)
            face_alignment_net.eval()

        if face_detection_net is None:
            #face detection 
            face_detector_model = r"./models/s3fd-619a316812.pth"
            face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)
    
    #localize the face in the video 
    # localize_face = 0 -> Face is localized at a single frame in the video (the middle frame)
    # localize_face = -1 -> Face is localized at each frame of the video
    # localize_face = n -> face is localized every n frames 

    # we will start by localizing the face in the middel of the video, if additional information is needed 
    # then will be added as required

    
    
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    video_handler = cv2.VideoCapture(color_file)  # read the video
    num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_handler.get(cv2.CAP_PROP_FPS))
    video_handler.set(cv2.CAP_PROP_POS_FRAMES, num_frames//2)

    success, image = video_handler.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w,_=image.shape

    if success: 
        detected_faces = face_detection_net.detect_from_image(image)
        for i, d in enumerate(detected_faces):

            if d[4]>0.9:
                if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough   

                    found_face = d

                    center = torch.FloatTensor(
                        [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                    center[1] = center[1] - (d[3] - d[1]) * 0.12
                    scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                    if fix_head_position:
                        try:
                            d[3]=d[3]+fix_head_position
                            center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                            center[1] = center[1] - (d[3] - d[1]) * 0.12
                            scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                        except:
                            pass
                        
    video_handler.release()


    data_video = pd.read_csv(cvs_frame_info, index_col=0)
    try:
        true_frame_number = data_video['BAG_Frame_Number'].tolist()
    except:
        true_frame_number = data_video['Actual_Frame_Number'].tolist() 
        
    time_st = data_video['Frame_Time_Stamp'].tolist()
    video_frame_number = data_video['Frame_Number_in_Video'].tolist()


    #create a dataframe that will store all the information 
    df_cols = ["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
    for i in range(0,68):
        num=str(i)
        xx = 'landmark_'+num+'_x'
        yy = 'landmark_'+num+'_y'
        df_cols.append(xx)
        df_cols.append(yy)

    LandmarkDataFrame = pd.DataFrame(columns = df_cols)

    # re-position the video handler at the first frame and start going frame by frame
    video_handler = cv2.VideoCapture(color_file)  # read the video
    k = 0
    success = True
    images = []
    centers = []
    scales = []
    Faces = []

    Frame_num = []

    while success:
        current_frame_num  = int(video_handler.get(cv2.CAP_PROP_POS_FRAMES))
        success, image = video_handler.read()
        if success:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if localize_face == 0:
                #do not localize the face, use previous info 
                pass 
            elif localize_face == -1 :
                #localize the face at each frame, upd
                update_detected_face = face_detection_net.detect_from_image(image)
                for i, d in enumerate(update_detected_face):

                    if d[4]>=0.9:

                        if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough   

                            #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
                            found_face = d
                            center = torch.FloatTensor(
                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                            center[1] = center[1] - (d[3] - d[1]) * 0.12
                            scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                            if fix_head_position:
                                try:
                                    d[3]=d[3]+fix_head_position
                                    center = torch.FloatTensor(
                                            [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                    center[1] = center[1] - (d[3] - d[1]) * 0.12
                                    scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                except:
                                    pass
            else:
                #localize face in the first frame and every n frame 
                if k == 0:
                    update_detected_face = face_detection_net.detect_from_image(image)
                    for i, d in enumerate(update_detected_face):
                        if d[4]>=0.9:
                            #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
                            if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough

                                found_face = d
                                center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                center[1] = center[1] - (d[3] - d[1]) * 0.12
                                scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                                if fix_head_position:
                                    try:
                                        d[3]=d[3]+fix_head_position
                                        center = torch.FloatTensor(
                                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                        center[1] = center[1] - (d[3] - d[1]) * 0.12
                                        scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                    except:
                                        pass

                #only update every n frames
                if (k+1)%localize_face == 0:
                    update_detected_face = face_detection_net.detect_from_image(image)
                    for i, d in enumerate(update_detected_face):
                        if d[4]>=0.9:
                            #do we trust the face localizer, if yes (>0.8) then update the bounding box,
                            if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough
                                found_face = d
                                center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                center[1] = center[1] - (d[3] - d[1]) * 0.12
                                scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                                if fix_head_position:
                                    try:
                                        d[3]=d[3]+fix_head_position
                                        center = torch.FloatTensor(
                                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                        center[1] = center[1] - (d[3] - d[1]) * 0.12
                                        scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                    except:
                                        pass

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float()
            inp = inp.to(device)
            inp.div_(255)

            images.append(inp)
            centers.append(center)
            scales.append(scale)
            Faces.append(found_face)

            if (len(images)==batch_size) or (k==num_frames-1):

                out = face_alignment_net(torch.stack(images))[-1].detach() 
                pts, pts_img_all = get_preds_fromhm_subpixel(out, centers, scales)

                for n,(pts_img, face) in enumerate(zip(pts_img_all.detach().numpy(), Faces)):

                    pts_img = pts_img.reshape(68, 2)



                    # Store everything in a dataframe
                    datus = []
                    #datus.append(int(k)+1+n-batch_size+1)  # frame number in the color_only video 

                    datus.append(face[0])  #top
                    datus.append(face[1])  #left
                    datus.append(face[2])  #bottom
                    datus.append(face[3])  #right

                    all_landmarks = pts_img
                    for x,y in all_landmarks:
                        datus.append(x), datus.append(y)  #x and y position of each landmark

                    LandmarkDataFrame = LandmarkDataFrame.append(pd.Series(datus,index = df_cols), 
                                           ignore_index = True)

                images = []
                centers = []
                scales = []
                Faces = []

                
            Frame_num.append(current_frame_num)           
            k+=1    
 


    #add time to landmarks 
    
    
    LandmarkDataFrame.insert(loc=0, column="Video_Frame_number", value=video_frame_number)
    LandmarkDataFrame.insert(loc=1, column='Time_Stamp (s)', value=time_st)
    
                             

#     landmark_file = BAG_File[:-4]+'_landmarks.csv'
#     LandmarkDataFrame.to_csv(landmark_file) 
    
    base, file = os.path.split(BAG_File)
    base = sufix
    landmark_file = os.path.join(base,file[:-4]+'_landmarks.csv')
    LandmarkDataFrame.to_csv(landmark_file)



    #print('Success getting facial landmakrs')
    return landmark_file


def smooth_landmarks(landmark_file, color_file, create_video=False, sufix = 'landmarks'):
    
    LandmarkDataFrame = pd.read_csv(landmark_file, index_col=0)
    b, a = signal.bessel(2 ,0.1)
    windowlength=5
    for i in range(68):
        num=str(i)
        xx = LandmarkDataFrame['landmark_'+num+'_x'].values
        xx_med = signal.medfilt(xx,kernel_size=windowlength)
    #     mod_xx = sm.tsa.statespace.SARIMAX(xx, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
    #     res_xx = mod_xx.fit()
    #     predict_xx = res_xx.get_prediction(end=mod_xx.nobs +0-1)
    #     predict_xx_out = predict_xx.predicted_mean
    #     predict_xx_out[0] = xx[0]


        yy = LandmarkDataFrame['landmark_'+num+'_y'].values
        yy_med = signal.medfilt(yy,kernel_size=windowlength)
    #     mod_yy = sm.tsa.statespace.SARIMAX(yy, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
    #     res_yy = mod_yy.fit()
    #     predict_yy = res_yy.get_prediction(end=mod_yy.nobs +0-1)
    #     predict_yy_out = predict_yy.predicted_mean
    #     predict_yy_out[0] = yy[0]

        LandmarkDataFrame['landmark_'+num+'_x'] = xx_med
        LandmarkDataFrame['landmark_'+num+'_y'] = yy_med

#     landmark_file = BAG_File[:-4]+'_landmarksFiltered.csv'
#     LandmarkDataFrame.to_csv(landmark_file)
    
#     base, file = os.path.split(BAG_File)
#     base = os.path.join(base,sufix)
    landmark_file = landmark_file[:-4]+'Filtered.csv'
    LandmarkDataFrame.to_csv(landmark_file)

    if create_video:

        video_handler = cv2.VideoCapture(color_file)  # read the video
        num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS= int(video_handler.get(cv2.CAP_PROP_FPS))
        width = int(video_handler.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
        height = int(video_handler.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float


        color_file_landmark = landmark_file[:-4]+'.avi'

        video = VideoWriter(color_file_landmark, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(FPS), (width,height))
        video_handler = cv2.VideoCapture(color_file)  # read the video
        success = True
        k=0
        for k in range(int(num_frames)):
            try: 
                success, image = video_handler.read()

                frame_number=k
                frame_information = LandmarkDataFrame.loc[LandmarkDataFrame['Video_Frame_number'] == frame_number].values
                shape = np.array([frame_information[0][6:]])
                shape = np.reshape(shape.astype(np.int), (-1, 2))
                for (x, y) in shape:
                    if x is np.NaN:
                        continue
                    else:
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                frame_to_save = image
                video.write(frame_to_save)
            except:
                print(k)
            #k +=1


        video.release()

#         print('done -- please review video')
    return landmark_file


def get3dlandmarks(BAG_File, info_file, csv_file, sufix = 'landmarks'):

    DF_landmarks = pd.read_csv(csv_file, index_col=0)

    DF_info = pd.read_csv(info_file, index_col=0)


    # create dataframe to store information about 3d position of landmarks
    df_cols_p1 = ["Video_Frame_number", 'Time_Stamp (s)']
    for i in range(0, 68):
        num = str(i)
        xx = 'landmark_' + num
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)

    df_cols_p2 = ["",""]
    for i in range(0, 68):
        df_cols_p2.append("x")
        df_cols_p2.append("y")
        df_cols_p2.append("z")

    header = [np.array(df_cols_p1),
              np.array(df_cols_p2)]

    DF_3dpositions = pd.DataFrame(columns=header)

    DF_3dpositions['Time_Stamp (s)'] = DF_landmarks['Time_Stamp (s)']
    DF_3dpositions['Video_Frame_number'] = DF_landmarks['Video_Frame_number']

    # start the process of extracting the video information for each video
    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, BAG_File, repeat_playback=False)

    config.enable_all_streams()
    profile = pipeline.start(config)

    # create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # inform the device that this is not live streaming from camera
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    duration = playback.get_duration()

    # fill holes in the depth information (based on this example: https://nbviewer.jupyter.org/github/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 3)

    true_frame_number = []
    frame_number = []
    time_st = []

    num_frame = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(100)

            this_frame = frames.get_frame_number()  # get frame number        
            landmarks=None
            # verify that we have landmarks for this particular frame

            try:
                vid_frame = DF_info.Frame_Number_in_Video.loc[DF_info["BAG_Frame_Number"]==this_frame].values[0] 

                index_ = DF_info.index[DF_info['BAG_Frame_Number']==this_frame]

                landmarks = DF_landmarks.iloc[index_].values[0][6:]   
                landmarks = landmarks.astype('float').reshape(-1, 2)
                aligned_frames = align.process(frames)

                # take color and depth from frame, if any to these is not available then skip the frame
                aligned_frames = align.process(frames)

                # take color and depth from frame, if any to these is not available then skip the frame
                aligned_depth = aligned_frames.get_depth_frame()
                aligned_color = aligned_frames.get_color_frame()

                # validate that both frames are available
                if not aligned_depth or not aligned_color:
                    continue

                #time_stamp = frames.get_timestamp()
                #true_frame_number.append(frames.get_frame_number())
                #time_st.append(time_stamp)
                #frame_number.append(num_frame)

                # Intrinsics & Extrinsics
                depth_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
                color_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
                depth_to_color_extrin = aligned_depth.profile.get_extrinsics_to(aligned_color.profile)

                aligned_filtered_depth = spatial.process(aligned_depth)
                depth_frame_array = np.asanyarray(aligned_filtered_depth.as_frame().get_data())
                depth_frame_array = depth_frame_array * depth_scale

                coords = []

                for (c,
                     r) in landmarks:  # landmarks provide the x,y position of each landmark. x are columns and y are rows in the figure
                    # depth_value = depth_frame.get_distance(int(c),int(r))
                    # x,y,z = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(c), int(r)], depth_value)
                    try:
                        depth_value = depth_frame_array[int(r), int(c)]
                    except IndexError:
                        h,w = depth_frame_array.shape
                        if r>w-1:r=h-1
                        if c>h-1:c=w-1
                        depth_value = depth_frame_array[int(r), int(c)]

                    z = depth_value
                    x = z * ((c - depth_intrin.ppx) / depth_intrin.fx)
                    y = z * ((r - depth_intrin.ppy) / depth_intrin.fy)
                    coords.append(x), coords.append(y), coords.append(z)

                DF_3dpositions.loc[index_,2:] =coords
            except IndexError:
                error= 'Error file:' + BAG_File.split(os.path.sep)[-1] +'\n' + str(this_frame)+' does not exists in color video'
                print(error)


    except RuntimeError:
        pass
    finally:
        pipeline.stop()
        
        
    landmark_file = csv_file[:-4]+'3D.csv'
    DF_3dpositions.to_csv(landmark_file)



def estimate_3dlandmarks(csv_file,
                         device,
                         color_file,
                         sufix = 'landmarks',
                         face_detection_net=None,
                         depth_network=None):
    import face_alignment.utils as utils

    if  (depth_network is None) or (face_detection_net is None):

        from face_alignment import api as face_alignment
        from face_alignment.models import FAN, ResNetDepth
        from face_alignment.detection.sfd import sfd_detector


        if depth_network is None:
            depth_network = ResNetDepth()
            depth_model = r"./models/depth-2a464da4ea.pth.tar"
            sd = torch.load(depth_model, map_location=lambda storage, loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                                                v in sd['state_dict'].items()}

            depth_network.load_state_dict(depth_dict)
            depth_network.to(device)
            depth_network.eval()

        if face_detection_net is None:
            #face detection
            face_detector_model = r"./models/s3fd-619a316812.pth"
            face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)



    DF_landmarks = pd.read_csv(csv_file, index_col=0)

    # create dataframe to store information about 3d position of landmarks
    df_cols_p1 = [ "Video_Frame_number"]
    for i in range(0, 68):
        num = str(i)
        xx = 'landmark_' + num
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)

    df_cols_p2 = [""]
    for i in range(0, 68):
        df_cols_p2.append("x")
        df_cols_p2.append("y")
        df_cols_p2.append("z")

    header = [np.array(df_cols_p1),
              np.array(df_cols_p2)]

    DF_3dpositions = pd.DataFrame(columns=header)

    time_st = DF_landmarks['Time_Stamp (s)'].tolist()

    # re-position the video handler at the first frame and start going frame by frame
    video_handler = cv2.VideoCapture(color_file)  # read the video
    num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
    k = 0

    bbox_top_x = DF_landmarks['bbox_top_x'].tolist()
    bbox_top_y = DF_landmarks['bbox_top_y'].tolist()
    bbox_bottom_x = DF_landmarks['bbox_bottom_x'].tolist()
    bbox_bottom_y = DF_landmarks['bbox_bottom_y'].tolist()

    success = True
    for k in range(num_frames):
        current_frame_num = int(video_handler.get(cv2.CAP_PROP_POS_FRAMES))
        success, image = video_handler.read()
        if success:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            d = torch.zeros(4)
            d[0] = bbox_top_x[current_frame_num]
            d[1] = bbox_top_y[current_frame_num]
            d[2] = bbox_bottom_x[current_frame_num]
            d[3] = bbox_bottom_y[current_frame_num]

            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()
            inp = inp.to(device)
            inp.div_(255).unsqueeze_(0)

            landmarks = DF_landmarks.loc[DF_landmarks['Video_Frame_number'] == k ].values

            if len(landmarks) > 0:
                # continue only if landmarks for the frame are avaliable
                landmarks = landmarks[0][6:]
                landmarks = landmarks.astype('float').reshape(-1, 2)

            heatmaps = torch.zeros((68, 256, 256))
            for i in range(68):
                transformed_landmarks = transform(landmarks[i], center, scale, 256, invert=False, integer=False)
                transformed_landmarks = transformed_landmarks+0.5
                heatmaps[i] = draw_gaussian(heatmaps[i], transformed_landmarks, 1, offset=True)

            heatmaps = heatmaps.to(device).unsqueeze_(0)

            depth_pred = depth_network(
                torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)

            depth_pred_corrected = depth_pred * (1.0 / (256.0 / (200.0 * scale)))
            depth_pred_corrected = depth_pred_corrected.numpy()

            preds = np.column_stack((landmarks, depth_pred_corrected))

            # Store everything in a dataframe
            datus = []
            datus.append(current_frame_num)  # frame number in the color_only video

#             datus.append(d[0].item())  # top
#             datus.append(d[1].item())  # left
#             datus.append(d[2].item())  # bottom
#             datus.append(d[3].item())  # right

            for x, y, z in preds:
                datus.append(x), datus.append(y), datus.append(z)  # x and y position of each landmark

            DF_3dpositions = DF_3dpositions.append(pd.Series(datus, index=header),
                                                   ignore_index=True)

            #k += 1

    DF_3dpositions.insert(loc=1, column='Time_Stamp (s)', value=time_st)

#     landmark_file = cvs_file[:-4]+'_3DPredicted.csv'
#     DF_3dpositions.to_csv(landmark_file)
    
    base, file = os.path.split(color_file)
    base = os.path.join(base,sufix)
    landmark_file = csv_file[:-4]+'3DPredicted.csv'
    DF_3dpositions.to_csv(landmark_file)
    
    
    
    
#### Functions that work only with videos -- No .bag files 

def find_landmarks_video(device,
                   color_file,
                   localize_face=0,
                   sufix = 'landmarks',
                   batch_size=10,
                   fix_head_position=None,
                   face_alignment_net=None,
                   face_detection_net=None):
    
    import face_alignment.utils as utils
    
    if (face_alignment_net is None) or (face_detection_net is None):       
        from face_alignment import api as face_alignment
        from face_alignment.models import FAN
        from face_alignment.detection.sfd import sfd_detector

        def load_weights(model, filename):
            sd = torch.load(filename, map_location=lambda storage, loc: storage)
            names = set(model.state_dict().keys())
            for n in list(sd.keys()): 
                if n not in names and n+'_raw' in names:
                    if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
                    del sd[n]
            model.load_state_dict(sd)
            
        #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if face_alignment_net is None:
            face_alignment_model = r"./models/2DFAN4-11f355bf06.pth.tar"    
            #Face alignement
            network_size = 4
            face_alignment_net = FAN(network_size)
            load_weights(face_alignment_net,face_alignment_model)
            face_alignment_net.to(device)
            face_alignment_net.eval()

        if face_detection_net is None:
            #face detection 
            face_detector_model = r"./models/s3fd-619a316812.pth"
            face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)
    
    #localize the face in the video 
    # localize_face = 0 -> Face is localized at a single frame in the video (the middle frame)
    # localize_face = -1 -> Face is localized at each frame of the video
    # localize_face = n -> face is localized every n frames 

    # we will start by localizing the face in the middel of the video, if additional information is needed 
    # then will be added as required
    
    video_handler = cv2.VideoCapture(color_file)  # read the video
    num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(video_handler.get(cv2.CAP_PROP_FPS))
    video_handler.set(cv2.CAP_PROP_POS_FRAMES, num_frames//2)

    success, image = video_handler.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w,_=image.shape

    if success: 
        detected_faces = face_detection_net.detect_from_image(image)
        for i, d in enumerate(detected_faces):

            if d[4]>0.8:
                if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough   

                    found_face = d

                    center = torch.FloatTensor(
                        [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                    center[1] = center[1] - (d[3] - d[1]) * 0.12
                    scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                    if fix_head_position:
                        try:
                            d[3]=d[3]+fix_head_position
                            center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                            center[1] = center[1] - (d[3] - d[1]) * 0.12
                            scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                        except:
                            pass


    video_handler.release()


    #create a dataframe that will store all the information 
    df_cols = ["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
    for i in range(0,68):
        num=str(i)
        xx = 'landmark_'+num+'_x'
        yy = 'landmark_'+num+'_y'
        df_cols.append(xx)
        df_cols.append(yy)

    LandmarkDataFrame = pd.DataFrame(columns = df_cols)

    timestamps = []
    # re-position the video handler at the first frame and start going frame by frame
    video_handler = cv2.VideoCapture(color_file)  # read the video
    k = 0
    success = True

    images = []
    centers = []
    scales = []
    Faces = []

    Frame_num = []

    #while success:
    for k in range(int(num_frames)):

        current_frame_num = int(video_handler.get(cv2.CAP_PROP_POS_FRAMES))
        current_time_stamp = video_handler.get(cv2.CAP_PROP_POS_MSEC)/1000
        success, image = video_handler.read()
        if success:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if localize_face == 0:
                #do not localize the face, use previous info 
                pass 
            elif localize_face == -1 :
                #localize the face at each frame, upd
                update_detected_face = face_detection_net.detect_from_image(image)
                for i, d in enumerate(update_detected_face):

                    if d[4]>=0.8:

                        if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough   

                            #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
                            found_face = d
                            center = torch.FloatTensor(
                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                            center[1] = center[1] - (d[3] - d[1]) * 0.12
                            scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                            if fix_head_position:
                                try:
                                    d[3]=d[3]+fix_head_position
                                    center = torch.FloatTensor(
                                            [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                    center[1] = center[1] - (d[3] - d[1]) * 0.12
                                    scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                except:
                                    pass
            else:
                #localize face in the first frame and every n frame 
                if k == 0:
                    update_detected_face = face_detection_net.detect_from_image(image)
                    for i, d in enumerate(update_detected_face):
                        if d[4]>=0.8:
                            #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
                            if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough

                                found_face = d
                                center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                center[1] = center[1] - (d[3] - d[1]) * 0.12
                                scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                                if fix_head_position:
                                    try:
                                        d[3]=d[3]+fix_head_position
                                        center = torch.FloatTensor(
                                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                        center[1] = center[1] - (d[3] - d[1]) * 0.12
                                        scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                    except:
                                        pass


                #only update every n frames
                if (k+1)%localize_face == 0:
                    update_detected_face = face_detection_net.detect_from_image(image)
                    for i, d in enumerate(update_detected_face):
                        if d[4]>=0.8:
                            #do we trust the face localizer, if yes (>0.8) then update the bounding box,
                            if abs(d[2]-d[0])>h*(5/100): #verify that the face is big enough
                                found_face = d
                                center = torch.FloatTensor(
                                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                center[1] = center[1] - (d[3] - d[1]) * 0.12
                                scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

                                if fix_head_position:
                                    try:
                                        d[3]=d[3]+fix_head_position
                                        center = torch.FloatTensor(
                                                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                                        center[1] = center[1] - (d[3] - d[1]) * 0.12
                                        scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale                
                                    except:
                                        pass


            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float()
            inp = inp.to(device)
            inp.div_(255)

            images.append(inp)
            centers.append(center)
            scales.append(scale)
            Faces.append(found_face)

            if (len(images)==batch_size) or (k==num_frames-1):

                out = face_alignment_net(torch.stack(images))[-1].detach() 
                pts, pts_img_all = get_preds_fromhm_subpixel(out, centers, scales)

                for n,(pts_img, face) in enumerate(zip(pts_img_all.detach().numpy(), Faces)):

                    pts_img = pts_img.reshape(68, 2)



                    # Store everything in a dataframe
                    datus = []

                    datus.append(face[0])  #top
                    datus.append(face[1])  #left
                    datus.append(face[2])  #bottom
                    datus.append(face[3])  #right

                    all_landmarks = pts_img
                    for x,y in all_landmarks:
                        datus.append(x), datus.append(y)  #x and y position of each landmark

                    LandmarkDataFrame = LandmarkDataFrame.append(pd.Series(datus,index = df_cols), 
                                           ignore_index = True)

                images = []
                centers = []
                scales = []
                Faces = []

        Frame_num.append(current_frame_num)           
        timestamps.append(current_time_stamp)     
        #k+=1     




    #add time to landmarks 
    LandmarkDataFrame.insert(loc=0, column='Video_Frame_number', value=Frame_num[0:len(LandmarkDataFrame)])
    LandmarkDataFrame.insert(loc=1, column='Time_Stamp (s)', value=timestamps[0:len(LandmarkDataFrame)])
    video_handler.release()

    base, file = os.path.split(color_file)
    base = os.path.join(base,sufix)
    landmark_file = os.path.join(base,file[:-10]+'_landmarks.csv')
    LandmarkDataFrame.to_csv(landmark_file)
    
    return landmark_file


#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_fps = int(video_handler.get(cv2.CAP_PROP_FPS))
#     video_handler.set(cv2.CAP_PROP_POS_FRAMES, num_frames//2)

#     success, image = video_handler.read()
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if success: 
#         detected_faces = face_detection_net.detect_from_image(image)
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#     video_handler.release()

#     # at this point we have the position of the face in the mid frame. Let's use that info

#     #extend the face bounding box to improve localization
#     if fix_head_position:
#         detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale



#     #create a dataframe that will store all the information 
#     df_cols = ["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0,68):
#         num=str(i)
#         xx = 'landmark_'+num+'_x'
#         yy = 'landmark_'+num+'_y'
#         df_cols.append(xx)
#         df_cols.append(yy)

#     LandmarkDataFrame = pd.DataFrame(columns = df_cols)

#     timestamps = []
#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     Found_Face = True
#     k = 0
#     success = True

#     images = []
#     centers = []
#     scales = []
#     Faces = []

#     Frame_num = []

#     while success:
#         current_frame_num = int(video_handler.get(cv2.CAP_PROP_POS_FRAMES))
#         current_time_stamp = video_handler.get(cv2.CAP_PROP_POS_MSEC)/1000
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             if localize_face == 0:
#                 #do not localize the face, use previous info 
#                 pass 
#             elif localize_face == -1 :
#                 #localize the face at each frame, upd
#                 update_detected_face = face_detection_net.detect_from_image(image)
#                 for i, d in enumerate(update_detected_face):

#                     if d[4]>=0.8:
#                         #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                         detected_faces = update_detected_face
#                         center = torch.FloatTensor(
#                             [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                         center[1] = center[1] - (d[3] - d[1]) * 0.12
#                         scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#                         if fix_head_position:
#                             detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#                             for i, d in enumerate(detected_faces):
#                                 center = torch.FloatTensor(
#                                     [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                 center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                 scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#             else:
#                 #localize face in the first frame and every n frame 
#                 if k == 0:
#                     update_detected_face = face_detection_net.detect_from_image(image)
#                     for i, d in enumerate(update_detected_face):
#                         if d[4]>=0.8:
#                             #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                             detected_faces = update_detected_face
#                             center = torch.FloatTensor(
#                                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                             center[1] = center[1] - (d[3] - d[1]) * 0.12
#                             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#                             if fix_head_position:
#                                 detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#                                 for i, d in enumerate(detected_faces):
#                                     center = torch.FloatTensor(
#                                         [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                     center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                     scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale


#                 #only update every n frames
#                 if (k+1)%localize_face == 0:
#                     update_detected_face = face_detection_net.detect_from_image(image)
#                     for i, d in enumerate(update_detected_face):
#                         if d[4]>=0.8:
#                             #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                             detected_faces = update_detected_face
#                             center = torch.FloatTensor(
#                                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                             center[1] = center[1] - (d[3] - d[1]) * 0.12
#                             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#                             if fix_head_position:
#                                 try: 
#                                     detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#                                     for i, d in enumerate(detected_faces):
#                                         center = torch.FloatTensor(
#                                             [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                         center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                         scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#                                 except:
#                                     pass






#             inp = crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                         (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255)

#             images.append(inp)
#             centers.append(center)
#             scales.append(scale)
#             Faces.append(detected_faces)

#             if (len(images)==batch_size) or (k==num_frames-1):

#                 out = face_alignment_net(torch.stack(images))[-1].detach() 
#                 pts, pts_img_all = get_preds_fromhm_subpixel(out, centers, scales)

#                 for n,(pts_img, face) in enumerate(zip(pts_img_all.detach().numpy(), Faces)):

#                     pts_img = pts_img.reshape(68, 2)



#                     # Store everything in a dataframe
#                     datus = []

#                     datus.append(face[0][0])  #top
#                     datus.append(face[0][1])  #left
#                     datus.append(face[0][2])  #bottom
#                     datus.append(face[0][3])  #right

#                     all_landmarks = pts_img
#                     for x,y in all_landmarks:
#                         datus.append(x), datus.append(y)  #x and y position of each landmark

#                     LandmarkDataFrame = LandmarkDataFrame.append(pd.Series(datus,index = df_cols), 
#                                            ignore_index = True)

#                 images = []
#                 centers = []
#                 scales = []
#                 Faces = []

#         Frame_num.append(current_frame_num)           
#         timestamps.append(current_time_stamp)     
#         k+=1     




#     #add time to landmarks 
#     LandmarkDataFrame.insert(loc=0, column='Video_Frame_number', value=Frame_num[0:len(LandmarkDataFrame)])
#     LandmarkDataFrame.insert(loc=1, column='Time_Stamp (s)', value=timestamps[0:len(LandmarkDataFrame)])
#     video_handler.release()
    
#     base, file = os.path.split(color_file)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-10]+'_landmarks.csv')
#     LandmarkDataFrame.to_csv(landmark_file)   

#     return landmark_file

    


def smooth_landmarks_video(landmark_file, color_file, create_video=True, sufix = 'landmarks'):
    LandmarkDataFrame = pd.read_csv(landmark_file, index_col=0)
    #b, a = signal.bessel(2 ,0.1)
    windowlength=5
    for i in range(68):
        num=str(i)
        xx = LandmarkDataFrame['landmark_'+num+'_x'].values
        xx_med = signal.medfilt(xx,kernel_size=windowlength)
    #     mod_xx = sm.tsa.statespace.SARIMAX(xx, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
    #     res_xx = mod_xx.fit()
    #     predict_xx = res_xx.get_prediction(end=mod_xx.nobs +0-1)
    #     predict_xx_out = predict_xx.predicted_mean
    #     predict_xx_out[0] = xx[0]


        yy = LandmarkDataFrame['landmark_'+num+'_y'].values
        yy_med = signal.medfilt(yy,kernel_size=windowlength)
    #     mod_yy = sm.tsa.statespace.SARIMAX(yy, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
    #     res_yy = mod_yy.fit()
    #     predict_yy = res_yy.get_prediction(end=mod_yy.nobs +0-1)
    #     predict_yy_out = predict_yy.predicted_mean
    #     predict_yy_out[0] = yy[0]

        LandmarkDataFrame['landmark_'+num+'_x'] = xx_med
        LandmarkDataFrame['landmark_'+num+'_y'] = yy_med

        
    landmark_file = landmark_file[:-4] + 'Filtered.csv'
    LandmarkDataFrame.to_csv(landmark_file)
    
    
    if create_video:

        video_handler = cv2.VideoCapture(color_file)  # read the video
        num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS= int(video_handler.get(cv2.CAP_PROP_FPS))
        width = int(video_handler.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
        height = int(video_handler.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float


        color_file_landmark = landmark_file[:-4]+'.avi'

        video = VideoWriter(color_file_landmark, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(FPS), (width,height))
        video_handler = cv2.VideoCapture(color_file)  # read the video
        success = True
        k=0
        for k in range(int(num_frames)):
            success, image = video_handler.read()

            frame_number=k
            frame_information = LandmarkDataFrame.loc[LandmarkDataFrame['Video_Frame_number'] == frame_number].values
            shape = np.array([frame_information[0][6:]])
            shape = np.reshape(shape.astype(np.int), (-1, 2))
            for (x, y) in shape:
                if x is np.NaN:
                    continue
                else:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            frame_to_save = image
            video.write(frame_to_save)
            k +=1


        video.release()

    return landmark_file
        
        
# def estimate_3dlandmarks_video(cvs_file,
#                          device,
#                          color_file,
#                          sufix = 'landmarks',
#                          face_detection_net=None,
#                          depth_network=None):
#     import face_alignment.utils as utils

#     if  (depth_network is None) or (face_detection_net is None):

#         from face_alignment import api as face_alignment
#         from face_alignment.models import FAN, ResNetDepth
#         from face_alignment.detection.sfd import sfd_detector


#         if depth_network is None:
#             depth_network = ResNetDepth()
#             depth_model = r"./models/depth-2a464da4ea.pth.tar"
#             sd = torch.load(depth_model, map_location=lambda storage, loc: storage)
#             depth_dict = {
#                 k.replace('module.', ''): v for k,
#                                                 v in sd['state_dict'].items()}

#             depth_network.load_state_dict(depth_dict)
#             depth_network.to(device)
#             depth_network.eval()

#         if face_detection_net is None:
#             #face detection
#             face_detector_model = r"./models/s3fd-619a316812.pth"
#             face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)



#     DF_landmarks = pd.read_csv(cvs_file, index_col=0)

#     # create dataframe to store information about 3d position of landmarks
#     df_cols_p1 = ["Video_Frame_number", "bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0, 68):
#         num = str(i)
#         xx = 'landmark_' + num
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)

#     df_cols_p2 = ["", "", "", "", ""]
#     for i in range(0, 68):
#         df_cols_p2.append("x")
#         df_cols_p2.append("y")
#         df_cols_p2.append("z")

#     header = [np.array(df_cols_p1),
#               np.array(df_cols_p2)]

#     DF_3dpositions = pd.DataFrame(columns=header)

#     time_st = DF_landmarks['Time_Stamp (s)'].tolist()

#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     k = 0

#     bbox_top_x = DF_landmarks['bbox_top_x'].tolist()
#     bbox_top_y = DF_landmarks['bbox_top_y'].tolist()
#     bbox_bottom_x = DF_landmarks['bbox_bottom_x'].tolist()
#     bbox_bottom_y = DF_landmarks['bbox_bottom_y'].tolist()

#     success = True
#     while success:
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             d = torch.zeros(4)
#             d[0] = bbox_top_x[k]
#             d[1] = bbox_top_y[k]
#             d[2] = bbox_bottom_x[k]
#             d[3] = bbox_bottom_y[k]

#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#             inp = utils.crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                 (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255).unsqueeze_(0)

#             landmarks = DF_landmarks.loc[DF_landmarks['Video_Frame_number'] == k + 1].values

#             if len(landmarks) > 0:
#                 # continue only if landmarks for the frame are avaliable
#                 landmarks = landmarks[0][6:]
#                 landmarks = landmarks.astype('float').reshape(-1, 2)

#             heatmaps = torch.zeros((68, 256, 256))
#             for i in range(68):
#                 transformed_landmarks = transform(landmarks[i] + 1, center, scale, 256, invert=False)
#                 heatmaps[i] = draw_gaussian(heatmaps[i], transformed_landmarks, 2)

#             heatmaps = heatmaps.to(device).unsqueeze_(0)

#             depth_pred = depth_network(
#                 torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)

#             depth_pred_corrected = depth_pred * (1.0 / (256.0 / (200.0 * scale)))
#             depth_pred_corrected = depth_pred_corrected.numpy()

#             preds = np.column_stack((landmarks, depth_pred_corrected))

#             # Store everything in a dataframe
#             datus = []
#             datus.append(int(k) + 1)  # frame number in the color_only video

#             datus.append(d[0].item())  # top
#             datus.append(d[1].item())  # left
#             datus.append(d[2].item())  # bottom
#             datus.append(d[3].item())  # right

#             for x, y, z in preds:
#                 datus.append(x), datus.append(y), datus.append(z)  # x and y position of each landmark

#             DF_3dpositions = DF_3dpositions.append(pd.Series(datus, index=header),
#                                                    ignore_index=True)

#             k += 1

#     DF_3dpositions.insert(loc=1, column='Time_Stamp (s)', value=time_st)


#     base, file = os.path.split(color_file)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-4]+'_3DPredicted.csv')
#     DF_3dpositions.to_csv(landmark_file)
    
    
    
# functions to obtain depth information from .avi files (Andrea's code)
def get_3d(x,y,z):
    
    FOV_H_degree = 68;
    FOV_V_degree = 41.5;

    # field of view in radians
    fov_h = (FOV_H_degree * np.pi) / 180;
    fov_v = (FOV_V_degree * np.pi) / 180;
    
    
    # Calibrated
    f = [608.532, 609.732];
    
    # Calibrated
    cc = [319.5, 239.5] # x 
    
    return np.hstack([z * ((x - cc[0])/f[0]),z * ((y - cc[1])/f[1]),z])


def get3dlandmarks_video(filename_video, filename_csv):
    vid = cv2.VideoCapture(filename_video) 
    landmarksDataFrame = pd.read_csv(filename_csv, index_col=0)




    len_video = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    len_frames = len(landmarksDataFrame)
    
    h= int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w= int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    df_cols_p1 = ["Video_Frame_number", 'Time_Stamp (s)']
    for i in range(0, 68):
        num = str(i)
        xx = 'landmark_' + num
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)
        df_cols_p1.append(xx)

    df_cols_p2 = ["", ""]
    for i in range(0, 68):
        df_cols_p2.append("x")
        df_cols_p2.append("y")
        df_cols_p2.append("z")

    header = [np.array(df_cols_p1),
              np.array(df_cols_p2)]

    DF_3dpositions = pd.DataFrame(columns=header)

    DF_3dpositions[["Video_Frame_number", 'Time_Stamp (s)']]=landmarksDataFrame[["Video_Frame_number", 'Time_Stamp (s)']]


    if len_video != len_frames:
        tt = 'error File: '+filename_video.split(os.path.sep)[-1] + '\nframes video = ' +str(len_video)+' | Frames landmarks file = '+str(len_frames) +'\n'
        
        print(tt)
        if (abs(len_video-len_frames) > 1) :
            print('Video cannot be processed\n')
            
            return

 
    success, image = vid.read()
    frame_num = 0 
    while success:
    #for frame_num, image in enumerate(vid):
        info_frame = landmarksDataFrame.iloc[frame_num].values

        landmarks_ = info_frame[6:].reshape(-1,2)

        z = np.zeros((len(landmarks_),1))

        for k,(x,y) in enumerate(landmarks_):
                
            if x<0:x=0
            if x>w:x=w-1

            if y<0:y=0
            if y>h:y=h-1

            lsd = image[int(y),int(x),1]
            msd = image[int(y),int(x),0]

            dec_lsd = "{0:{fill}8b}".format(lsd, fill='0')
            dec_msd = "{0:{fill}8b}".format(msd, fill='0')


            temp_z = int(dec_msd+dec_lsd,2)

            if temp_z== 0 :
                    #get points from around the central point 
                    
                    
                ny_t = int(y)-4
                if ny_t<0: ny_t=0
                ny_b = int(y)+4
                if ny_b>h: ny_b=h-1

                nx_l = int(x)-4
                if nx_l<0: nx_l=0
                nx_r = int(x)+4
                if nx_r>w : nx_r=w-1


                temp = image[ny_t:ny_b,nx_l:nx_r,1].reshape(-1,1)
                temp = temp[temp!=0]
                if len(temp)>0:
                    lsd  = int(round(temp.mean()))
                else:
                    lsd = 0

                temp = image[ny_t:ny_b,nx_l:nx_r,0].reshape(-1,1)
                temp = temp[temp!=0]
                if len(temp)>0:
                    msd  = int(round(temp.mean()))
                else:
                    msd = 0

                dec_lsd = "{0:{fill}8b}".format(lsd, fill='0')
                dec_msd = "{0:{fill}8b}".format(msd, fill='0')

                temp_z = int(dec_msd+dec_lsd,2)

            z[k] = temp_z

        landmarks_3d = get_3d(landmarks_[:,0].reshape(-1,1),landmarks_[:,1].reshape(-1,1),z.reshape(-1,1))

        DF_3dpositions.iloc[frame_num,2:] = landmarks_3d.reshape(1,-1)[0]

        success, image = vid.read()
        frame_num += 1 
            
            
    DF_3dpositions.to_csv(filename_csv[:-4]+'3D.csv')
        
    return




#from here is the old version of this file, updated and modified on 12/03/2020 -- DLG

# import pyrealsense2 as rs
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from cv2 import VideoWriter, VideoWriter_fourcc
# import pandas as pd
# import torch
# from scipy import signal, ndimage, spatial
# import math


# def _gaussian_fast(size=3, sigma=0.25, amplitude=1., offset=[0., 0.], device='cpu'):
#     coordinates = torch.stack(torch.meshgrid(torch.arange(-size // 2 + 1. -offset[0], size // 2 + 1. -offset[0], step=1),
#                                    torch.arange(-size // 2 + 1. -offset[1], size // 2 + 1. -offset[1], step=1))).to(device)
#     coordinates = coordinates / (sigma * size)
#     gauss = amplitude * torch.exp(-(coordinates**2 / 2).sum(dim=0))
#     return gauss.permute(1, 0)


# def draw_gaussian(image, point, sigma, offset=False):
#     # Check if the gaussian is inside
#     ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
#     br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
#     if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
#         return image
#     size = 6 * sigma + 1
#     # g = torch.Tensor(_gaussian(size, offset=point%1) if offset else _gaussian(size), device=image.device)
#     g = _gaussian_fast(size, offset=point%1, device=image.device) if offset else _gaussian_fast(size, device=image.device)
#     g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
#     g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
#     img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
#     img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
#     assert (g_x[0] > 0 and g_y[1] > 0)
#     image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
#           ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
#     image[image > 1] = 1
#     return image

# def transform(point, center, scale, resolution, invert=False, integer=True):
#     """Generate and affine transformation matrix.
#     Given a set of points, a center, a scale and a targer resolution, the
#     function generates and affine transformation matrix. If invert is ``True``
#     it will produce the inverse transformation.
#     Arguments:
#         point {torch.tensor} -- the input 2D point
#         center {torch.tensor or numpy.array} -- the center around which to perform the transformations
#         scale {float} -- the scale of the face/object
#         resolution {float} -- the output resolution
#     Keyword Arguments:
#         invert {bool} -- define wherever the function should produce the direct or the
#         inverse transformation matrix (default: {False})
#     """
#     _pt = torch.ones(3)
#     _pt[0] = point[0]
#     _pt[1] = point[1]

#     h = 200.0 * scale
#     t = torch.eye(3)
#     t[0, 0] = resolution / h
#     t[1, 1] = resolution / h
#     t[0, 2] = resolution * (-center[0] / h + 0.5)
#     t[1, 2] = resolution * (-center[1] / h + 0.5)

#     if invert:
#         t = torch.inverse(t)

#     new_point = (torch.matmul(t, _pt))[0:2]

#     return new_point.int() if integer else new_point

# def create_target_heatmap(target_landmarks, centers, scales, size = 64):
#     """
#     Receives a batch of landmarks and returns a set of heatmaps for each set of 68 landmarks in the batch
#     :param target_landmarks: the batch is expected to have the dim (n x 68 x 2). Where n is the batch size
#     :return: returns a (n x 68 x 64 x 64) batch of heatmaps
#     """
#     heatmaps = np.zeros((target_landmarks.shape[0], 68, size, size), dtype=np.float32)
#     for i in range(heatmaps.shape[0]):
#         for p in range(68):
#             # Lua code from https://github.com/1adrianb/face-alignment-training/blob/master/dataset-images.lua:
#             # drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, 64), 1)
#             # Not sure why it adds 1 to each landmark before transform.
#             landmark_cropped_coor = transform(target_landmarks[i, p] + 1, centers[i], scales[i], 64, invert=False)
#             heatmaps[i, p] = draw_gaussian(heatmaps[i, p], landmark_cropped_coor + 1, 1)
#     return torch.tensor(heatmaps)


# def get_color_video(BAG_File):

#     pipeline = rs.pipeline()
#     config = rs.config()

#     rs.config.enable_device_from_file(config, BAG_File, repeat_playback=False)

#     config.enable_all_streams()
#     profile = pipeline.start(config)

#     # create alignment object
#     align_to = rs.stream.color
#     align = rs.align(align_to)

#     # inform the device that this is not live streaming from camera
#     playback = profile.get_device().as_playback()
#     playback.set_real_time(False)
#     duration = playback.get_duration()

#     true_frame_number = []
#     frame_number = []
#     time_st = []

#     num_frame = 0


#     Color_Frames = []#{}

#     try:
#         while True:
#             frames = pipeline.wait_for_frames(100)  #get frame from file 

#             this_frame = frames.get_frame_number()  #get frame number 

#             if (num_frame != 0) and (true_frame_number[-1] == this_frame): #verify that frame number is not repeated 
#                 #if frame number is repeated then replace the stored information 
#                 aligned_frames = align.process(frames)

#                 #take color and depth from frame, if any to these is not available then skip the frame
#                 aligned_depth = aligned_frames.get_depth_frame()
#                 aligned_color = aligned_frames.get_color_frame()

#                 # validate that both frames are available
#                 if not aligned_depth or not aligned_color:
#                     continue

#                 time_stamp = frames.get_timestamp()
#                 true_frame_number[-1] = frames.get_frame_number()
#                 time_st[-1] = time_stamp 

#                 # transform to np array

#                 color_data = np.asanyarray(aligned_color.as_frame().get_data(), dtype=np.int)
#                 #depth_data = np.asanyarray(aligned_depth.as_frame().get_data(), dtype=np.int)
#                 # adjust depth data in meters
#                 #depth_data *= depth_scale

#                 Color_Frames[-1] = color_data

#             else:
#                 #if frame number is not repeated then append the stored information 
#                 aligned_frames = align.process(frames)

#                 #take color and depth from frame, if any to these is not available then skip the frame
#                 aligned_depth = aligned_frames.get_depth_frame()
#                 aligned_color = aligned_frames.get_color_frame()

#                 # validate that both frames are available
#                 if not aligned_depth or not aligned_color:
#                     continue

#                 time_stamp = frames.get_timestamp()
#                 true_frame_number.append(frames.get_frame_number())
#                 time_st.append(time_stamp )

#                 # transform to np array

#                 color_data = np.asanyarray(aligned_color.as_frame().get_data(), dtype=np.int)
#                 #depth_data = np.asanyarray(aligned_depth.as_frame().get_data(), dtype=np.int)
#                 # adjust depth data in meters
#                 #depth_data *= depth_scale

#                 Color_Frames.append(color_data)
#                 #Depth_Frames.append(depth_data

#                 frame_number.append(num_frame)
#                 num_frame += 1

#     except RuntimeError:
#         pass
#     finally:
#         pipeline.stop()
        
#     duration_movie = duration.total_seconds()
#     FPS = num_frame/duration_movie
#     height, width,_ =  Color_Frames[0].shape

#     color_file = BAG_File[:-4]+'_color.avi'  
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  #for uncompresed AVI use (*'RGBA') or try  CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
#     video = VideoWriter(color_file,fourcc, int(FPS), (width,height))
#     #0x00000021
#     for k in range(num_frame):
#         frame_to_save = Color_Frames[k].astype('uint8')
#         video.write(frame_to_save)

#     video.release()    


#     cvs_frame_info = BAG_File[:-4]+'_frameInfoColor.csv'
#     df_cols = ['Actual_Frame_Number', 'Frame_Time_Stamp', 'Frame_Number_in_Video']
#     df = pd.DataFrame(columns=df_cols)
#     df['Actual_Frame_Number'] = true_frame_number
#     df['Frame_Time_Stamp'] = (np.array(time_st)-time_st[0])/1000
#     df['Frame_Number_in_Video'] = frame_number

#     df.to_csv(cvs_frame_info)
#     print('success reading BAG file')
    
#     return color_file, cvs_frame_info


# # ### This portion of the notebook takes the color information and finds the position of facial landmarks using FAN

# # #### Load models for face and facial landmarks localization

# # In[34]:


# def find_landmarks(BAG_File,
#                    device,
#                    color_file,
#                    cvs_frame_info,
#                    localize_face=0,
#                    sufix = 'landmarks',
#                    fix_head_position=True,
#                    face_alignment_net=None,
#                    face_detection_net=None):
    
#     import face_alignment.utils as utils
    
#     if (face_alignment_net is None) or (face_detection_net is None):       
#         from face_alignment import api as face_alignment
#         from face_alignment.models import FAN
#         from face_alignment.detection.sfd import sfd_detector

#         def load_weights(model, filename):
#             sd = torch.load(filename, map_location=lambda storage, loc: storage)
#             names = set(model.state_dict().keys())
#             for n in list(sd.keys()): 
#                 if n not in names and n+'_raw' in names:
#                     if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
#                     del sd[n]
#             model.load_state_dict(sd)
            
#         #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         if face_alignment_net is None:
#             face_alignment_model = r"./models/2DFAN4-11f355bf06.pth.tar"    
#             #Face alignement
#             network_size = 4
#             face_alignment_net = FAN(network_size)
#             load_weights(face_alignment_net,face_alignment_model)
#             face_alignment_net.to(device)
#             face_alignment_net.eval()

#         if face_detection_net is None:
#             #face detection 
#             face_detector_model = r"./models/s3fd-619a316812.pth"
#             face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)
    
#     #localize the face in the video 
#     # localize_face = 0 -> Face is localized at a single frame in the video (the middle frame)
#     # localize_face = -1 -> Face is localized at each frame of the video
#     # localize_face = n -> face is localized every n frames 

#     # we will start by localizing the face in the middel of the video, if additional information is needed 
#     # then will be added as required

    
    
#     #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_fps = int(video_handler.get(cv2.CAP_PROP_FPS))
#     video_handler.set(cv2.CAP_PROP_POS_FRAMES, num_frames//2)

#     success, image = video_handler.read()
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if success: 
#         detected_faces = face_detection_net.detect_from_image(image)
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#     video_handler.release()
    
#     # at this point we have the position of the face in the mid frame. Let's use that info
    
#     #extend the face bounding box to improve localization
#     if fix_head_position:
#         detected_faces[0][3] = detected_faces[0][3]+10
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale


#     data_video = pd.read_csv(cvs_frame_info, index_col=0)
#     true_frame_number = data_video['Actual_Frame_Number'].tolist()
#     time_st = data_video['Frame_Time_Stamp'].tolist()


#     #create a dataframe that will store all the information 
#     df_cols = ["BAG_Frame_number","Video_Frame_number", "bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0,68):
#         num=str(i)
#         xx = 'landmark_'+num+'_x'
#         yy = 'landmark_'+num+'_y'
#         df_cols.append(xx)
#         df_cols.append(yy)

#     LandmarkDataFrame = pd.DataFrame(columns = df_cols)

#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     k = 0
#     success = True
#     while success:
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             if localize_face == 0:
#                 #do not localize the face, use previous info 
#                 pass 
#             elif localize_face == -1 :
#                 #localize the face at each frame, upd
#                 update_detected_faces = face_detection_net.detect_from_image(image)
#                 for i, d in enumerate(update_detected_face):

#                     if d[4]>=0.8:
#                         #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                         # if not (<0.8) don't update the bounding box
#                         detected_faces = update_detected_face
#                         center = torch.FloatTensor(
#                             [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                         center[1] = center[1] - (d[3] - d[1]) * 0.12
#                         scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
                        
#                         if fix_head_position:
#                             detected_faces[0][3] = detected_faces[0][3]+40
#                             for i, d in enumerate(detected_faces):
#                                 center = torch.FloatTensor(
#                                     [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                 center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                 scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#             else:
#                 #only update every n frames
#                 if (k+1)%localize_face == 0:
#                     update_detected_face = face_detection_net.detect_from_image(image)
#                     for i, d in enumerate(update_detected_face):

#                         if d[4]>=0.8:
#                             #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                             # if not (<0.8) don't update the bounding box
#                             detected_faces = update_detected_face
#                             center = torch.FloatTensor(
#                                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                             center[1] = center[1] - (d[3] - d[1]) * 0.12
#                             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
                            
#                             if fix_head_position:
#                                 detected_faces[0][3] = detected_faces[0][3]+40
#                                 for i, d in enumerate(detected_faces):
#                                     center = torch.FloatTensor(
#                                         [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                     center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                     scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale


#             inp = utils.crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                         (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255).unsqueeze_(0)

#             out = face_alignment_net(inp)[-1].detach() #[-1] is to get the output of the last hourglass block
#             out = out.cpu()
#             pts, pts_img = utils.get_preds_fromhm(out, center, scale)

#             pts_img = pts_img.view(68, 2)



#             # Store everything in a dataframe
#             datus = []
#             datus.append(true_frame_number[int(k)])  #frame number provided by the .bag file 
#             datus.append(int(k)+1)  # frame number in the color_only video 

#             datus.append(detected_faces[0][0])  #top
#             datus.append(detected_faces[0][1])  #left
#             datus.append(detected_faces[0][2])  #bottom
#             datus.append(detected_faces[0][3])  #right

#             all_landmarks = pts_img.numpy()
#             for x,y in all_landmarks:
#                 datus.append(x), datus.append(y)  #x and y position of each landmark

#             LandmarkDataFrame = LandmarkDataFrame.append(pd.Series(datus,index = df_cols), 
#                                    ignore_index = True)

#             k +=1 


#     #add time to landmarks 
#     LandmarkDataFrame.insert(loc=1, column='Time_Stamp (s)', value=time_st)

# #     landmark_file = BAG_File[:-4]+'_landmarks.csv'
# #     LandmarkDataFrame.to_csv(landmark_file) 
    
#     base, file = os.path.split(BAG_File)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-4]+'_landmarks.csv')
#     LandmarkDataFrame.to_csv(landmark_file)



#     #print('Success getting facial landmakrs')
#     return landmark_file


# def smooth_landmarks(landmark_file, color_file, create_video=False, sufix = 'landmarks'):
    
#     LandmarkDataFrame = pd.read_csv(landmark_file, index_col=0)
#     b, a = signal.bessel(2 ,0.1)
#     windowlength=5
#     for i in range(68):
#         num=str(i)
#         xx = LandmarkDataFrame['landmark_'+num+'_x'].values
#         xx_med = signal.medfilt(xx,kernel_size=windowlength)
#     #     mod_xx = sm.tsa.statespace.SARIMAX(xx, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
#     #     res_xx = mod_xx.fit()
#     #     predict_xx = res_xx.get_prediction(end=mod_xx.nobs +0-1)
#     #     predict_xx_out = predict_xx.predicted_mean
#     #     predict_xx_out[0] = xx[0]


#         yy = LandmarkDataFrame['landmark_'+num+'_y'].values
#         yy_med = signal.medfilt(yy,kernel_size=windowlength)
#     #     mod_yy = sm.tsa.statespace.SARIMAX(yy, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
#     #     res_yy = mod_yy.fit()
#     #     predict_yy = res_yy.get_prediction(end=mod_yy.nobs +0-1)
#     #     predict_yy_out = predict_yy.predicted_mean
#     #     predict_yy_out[0] = yy[0]

#         LandmarkDataFrame['landmark_'+num+'_x'] = xx_med
#         LandmarkDataFrame['landmark_'+num+'_y'] = yy_med

# #     landmark_file = BAG_File[:-4]+'_landmarksFiltered.csv'
# #     LandmarkDataFrame.to_csv(landmark_file)
    
# #     base, file = os.path.split(BAG_File)
# #     base = os.path.join(base,sufix)
#     landmark_file = landmark_file[:-4]+'_Filtered.csv'
#     LandmarkDataFrame.to_csv(landmark_file)

#     if create_video:

#         video_handler = cv2.VideoCapture(color_file)  # read the video
#         num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
#         FPS= int(video_handler.get(cv2.CAP_PROP_FPS))
#         width = int(video_handler.get(3)) # float
#         height = int(video_handler.get(4)) # float

#         color_file_landmark = landmark_file[:-4]+'.mp4'

#         video = VideoWriter(color_file_landmark, -1, int(FPS), (width,height))
#         video_handler = cv2.VideoCapture(color_file)  # read the video
#         success = True
#         k=0
#         for k in range(int(num_frames)):
#             success, image = video_handler.read()

#             frame_number=k+1
#             frame_information = LandmarkDataFrame.loc[LandmarkDataFrame['Video_Frame_number'] == frame_number].values
#             shape = np.array([frame_information[0][7:]])
#             shape = np.reshape(shape.astype(np.int), (-1, 2))
#             for (x, y) in shape:
#                 if x is np.NaN:
#                     continue
#                 else:
#                     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

#             frame_to_save = image
#             video.write(frame_to_save)
#             k +=1

#         video.release()

# #         print('done -- please review video')
#         return landmark_file


# def get3dlandmarks(BAG_File,csv_file, sufix = 'landmarks'):

#     DF_landmarks = pd.read_csv(csv_file, index_col=0)


#     # create dataframe to store information about 3d position of landmarks
#     df_cols_p1 = ["BAG_Frame_number", "Video_Frame_number", ]
#     for i in range(0, 68):
#         num = str(i)
#         xx = 'landmark_' + num
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)

#     df_cols_p2 = ["", ""]
#     for i in range(0, 68):
#         df_cols_p2.append("x")
#         df_cols_p2.append("y")
#         df_cols_p2.append("z")

#     header = [np.array(df_cols_p1),
#               np.array(df_cols_p2)]

#     DF_3dpositions = pd.DataFrame(columns=header)

#     # start the process of extracting the video information for each video
#     pipeline = rs.pipeline()
#     config = rs.config()

#     rs.config.enable_device_from_file(config, BAG_File, repeat_playback=False)

#     config.enable_all_streams()
#     profile = pipeline.start(config)

#     # create alignment object
#     align_to = rs.stream.color
#     align = rs.align(align_to)

#     # Getting the depth sensor's depth scale (see rs-align example for explanation)
#     depth_sensor = profile.get_device().first_depth_sensor()
#     depth_scale = depth_sensor.get_depth_scale()

#     # inform the device that this is not live streaming from camera
#     playback = profile.get_device().as_playback()
#     playback.set_real_time(False)
#     duration = playback.get_duration()

#     # fill holes in the depth information (based on this example: https://nbviewer.jupyter.org/github/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb)
#     spatial = rs.spatial_filter()
#     spatial.set_option(rs.option.filter_magnitude, 2)
#     spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
#     spatial.set_option(rs.option.filter_smooth_delta, 20)
#     spatial.set_option(rs.option.holes_fill, 3)

#     true_frame_number = []
#     frame_number = []
#     time_st = []

#     num_frame = 0

#     try:
#         while True:
#             frames = pipeline.wait_for_frames(100)

#             this_frame = frames.get_frame_number()  # get frame number

#             # verify that we have landmarks for this particular frame
#             landmarks = DF_landmarks.loc[DF_landmarks['BAG_Frame_number'] == this_frame].values

#             # if there are not landmakrs then just ignore the frame

#             if len(landmarks) > 0:
#                 # continue only if landmarks for the frame are avaliable
#                 landmarks = landmarks[0][7:]
#                 landmarks = landmarks.astype('float').reshape(-1, 2)

#                 if (num_frame != 0) and (true_frame_number[-1] == this_frame):  # verify that frame number is not repeated
#                     # frame is repeated
#                     aligned_frames = align.process(frames)

#                     # take color and depth from frame, if any to these is not available then skip the frame
#                     aligned_depth = aligned_frames.get_depth_frame()
#                     aligned_color = aligned_frames.get_color_frame()

#                     # validate that both frames are available
#                     if not aligned_depth or not aligned_color:
#                         continue

#                     time_stamp = frames.get_timestamp()
#                     true_frame_number[-1] = frames.get_frame_number()
#                     time_st[-1] = time_stamp
#                     frame_number[-1] = num_frame

#                     # Intrinsics & Extrinsics
#                     depth_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
#                     color_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
#                     depth_to_color_extrin = aligned_depth.profile.get_extrinsics_to(aligned_color.profile)

#                     aligned_filtered_depth = spatial.process(aligned_depth)
#                     depth_frame_array = np.asanyarray(aligned_filtered_depth.get_data())
#                     depth_frame_array = depth_frame_array * depth_scale

#                     coords = []
#                     coords.append(frames.get_frame_number())
#                     coords.append(int(num_frame) + 1)

#                     for (c,
#                          r) in landmarks:  # landmarks provide the x,y position of each landmark. x are columns and y are rows in the figure
#                         # depth_value = depth_frame.get_distance(int(c),int(r))
#                         # x,y,z = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(c), int(r)], depth_value)
#                         depth_value = depth_frame_array[int(r), int(c)]
#                         z = depth_value
#                         x = z * ((c - depth_intrin.ppx) / depth_intrin.fx)
#                         y = z * ((r - depth_intrin.ppy) / depth_intrin.fy)
#                         coords.append(x), coords.append(y), coords.append(z)

#                     # DF_3dpositions = DF_3dpositions.append(pd.Series(coords,index = header), ignore_index = True)
#                     DF_3dpositions.iloc[-1] = pd.Series(coords, index=header)
#                 else:
#                     aligned_frames = align.process(frames)

#                     # take color and depth from frame, if any to these is not available then skip the frame
#                     aligned_depth = aligned_frames.get_depth_frame()
#                     aligned_color = aligned_frames.get_color_frame()

#                     # validate that both frames are available
#                     if not aligned_depth or not aligned_color:
#                         continue

#                     time_stamp = frames.get_timestamp()
#                     true_frame_number.append(frames.get_frame_number())
#                     time_st.append(time_stamp)
#                     frame_number.append(num_frame)

#                     # Intrinsics & Extrinsics
#                     depth_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
#                     color_intrin = aligned_depth.profile.as_video_stream_profile().intrinsics
#                     depth_to_color_extrin = aligned_depth.profile.get_extrinsics_to(aligned_color.profile)

#                     aligned_filtered_depth = spatial.process(aligned_depth)
#                     depth_frame_array = np.asanyarray(aligned_filtered_depth.as_frame().get_data())
#                     depth_frame_array = depth_frame_array * depth_scale

#                     coords = []
#                     coords.append(frames.get_frame_number())
#                     coords.append(int(num_frame) + 1)

#                     for (c,
#                          r) in landmarks:  # landmarks provide the x,y position of each landmark. x are columns and y are rows in the figure
#                         # depth_value = depth_frame.get_distance(int(c),int(r))
#                         # x,y,z = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(c), int(r)], depth_value)
#                         depth_value = depth_frame_array[int(r), int(c)]
#                         z = depth_value
#                         x = z * ((c - depth_intrin.ppx) / depth_intrin.fx)
#                         y = z * ((r - depth_intrin.ppy) / depth_intrin.fy)
#                         coords.append(x), coords.append(y), coords.append(z)

#                     DF_3dpositions = DF_3dpositions.append(pd.Series(coords, index=header), ignore_index=True)

#                     num_frame += 1

#     except RuntimeError:
#         pass
#     finally:
#         pipeline.stop()

#     # add time to 3d coordinates
#     DF_3dpositions.insert(loc=1, column='Time_Stamp (s)', value=(np.array(time_st) - time_st[0]) / 1000)
# #     landmarks_3D_file = BAG_File[:-4] + '_Landmarks3D.csv'
# #     DF_3dpositions.to_csv(landmarks_3D_file)
#     #save
#     base, file = os.path.split(BAG_File)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-4]+'_Landmarks3D.csv')
#     DF_3dpositions.to_csv(landmark_file)

#     # cvs_frame_info = BAG_File[:-4] + '_frameInfoDepth.csv'
#     # DF = pd.DataFrame()
#     # DF['Actual_Frame_Number'] = true_frame_number
#     # DF['Frame_Time_Stamp'] = (np.array(time_st) - time_st[0]) / 1000
#     # DF['Frame_Number_in_Video'] = frame_number
#     # DF.to_csv(cvs_frame_info)

#     #print('success getting 3d landmark')


# def estimate_3dlandmarks(cvs_file,
#                          device,
#                          color_file,
#                          sufix = 'landmarks',
#                          face_detection_net=None,
#                          depth_network=None):
#     import face_alignment.utils as utils

#     if  (depth_network is None) or (face_detection_net is None):

#         from face_alignment import api as face_alignment
#         from face_alignment.models import FAN, ResNetDepth
#         from face_alignment.detection.sfd import sfd_detector


#         if depth_network is None:
#             depth_network = ResNetDepth()
#             depth_model = r"./models/depth-2a464da4ea.pth.tar"
#             sd = torch.load(depth_model, map_location=lambda storage, loc: storage)
#             depth_dict = {
#                 k.replace('module.', ''): v for k,
#                                                 v in sd['state_dict'].items()}

#             depth_network.load_state_dict(depth_dict)
#             depth_network.to(device)
#             depth_network.eval()

#         if face_detection_net is None:
#             #face detection
#             face_detector_model = r"./models/s3fd-619a316812.pth"
#             face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)



#     DF_landmarks = pd.read_csv(cvs_file, index_col=0)

#     # create dataframe to store information about 3d position of landmarks
#     df_cols_p1 = ["BAG_Frame_number", "Video_Frame_number", "bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0, 68):
#         num = str(i)
#         xx = 'landmark_' + num
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)

#     df_cols_p2 = ["", "", "", "", "", ""]
#     for i in range(0, 68):
#         df_cols_p2.append("x")
#         df_cols_p2.append("y")
#         df_cols_p2.append("z")

#     header = [np.array(df_cols_p1),
#               np.array(df_cols_p2)]

#     DF_3dpositions = pd.DataFrame(columns=header)

#     true_frame_number = DF_landmarks['BAG_Frame_number'].tolist()
#     time_st = DF_landmarks['Time_Stamp (s)'].tolist()

#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     k = 0

#     bbox_top_x = DF_landmarks['bbox_top_x'].tolist()
#     bbox_top_y = DF_landmarks['bbox_top_y'].tolist()
#     bbox_bottom_x = DF_landmarks['bbox_bottom_x'].tolist()
#     bbox_bottom_y = DF_landmarks['bbox_bottom_y'].tolist()

#     success = True
#     while success:
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             d = torch.zeros(4)
#             d[0] = bbox_top_x[k]
#             d[1] = bbox_top_y[k]
#             d[2] = bbox_bottom_x[k]
#             d[3] = bbox_bottom_y[k]

#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#             inp = utils.crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                 (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255).unsqueeze_(0)

#             landmarks = DF_landmarks.loc[DF_landmarks['Video_Frame_number'] == k + 1].values

#             if len(landmarks) > 0:
#                 # continue only if landmarks for the frame are avaliable
#                 landmarks = landmarks[0][7:]
#                 landmarks = landmarks.astype('float').reshape(-1, 2)

#             heatmaps = torch.zeros((68, 256, 256))
#             for i in range(68):
#                 transformed_landmarks = transform(landmarks[i] + 1, center, scale, 256, invert=False)
#                 heatmaps[i] = draw_gaussian(heatmaps[i], transformed_landmarks, 2)

#             heatmaps = heatmaps.to(device).unsqueeze_(0)

#             depth_pred = depth_network(
#                 torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)

#             depth_pred_corrected = depth_pred * (1.0 / (256.0 / (200.0 * scale)))
#             depth_pred_corrected = depth_pred_corrected.numpy()

#             preds = np.column_stack((landmarks, depth_pred_corrected))

#             # Store everything in a dataframe
#             datus = []
#             datus.append(true_frame_number[int(k)])  # frame number provided by the .bag file
#             datus.append(int(k) + 1)  # frame number in the color_only video

#             datus.append(d[0].item())  # top
#             datus.append(d[1].item())  # left
#             datus.append(d[2].item())  # bottom
#             datus.append(d[3].item())  # right

#             for x, y, z in preds:
#                 datus.append(x), datus.append(y), datus.append(z)  # x and y position of each landmark

#             DF_3dpositions = DF_3dpositions.append(pd.Series(datus, index=header),
#                                                    ignore_index=True)

#             k += 1

#     DF_3dpositions.insert(loc=1, column='Time_Stamp (s)', value=time_st)

# #     landmark_file = cvs_file[:-4]+'_3DPredicted.csv'
# #     DF_3dpositions.to_csv(landmark_file)
    
#     base, file = os.path.split(color_file)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-10]+'_Landmarks3DPredicted.csv')
#     DF_3dpositions.to_csv(landmark_file)
    
    
    
    
# #### Functions that work only with videos -- No .bag files 

# def find_landmarks_video(device,
#                    color_file,
#                    localize_face=0,
#                    sufix = 'landmarks',
#                    fix_head_position=None,
#                    face_alignment_net=None,
#                    face_detection_net=None):
    
#     import face_alignment.utils as utils
    
#     if (face_alignment_net is None) or (face_detection_net is None):       
#         from face_alignment import api as face_alignment
#         from face_alignment.models import FAN
#         from face_alignment.detection.sfd import sfd_detector

#         def load_weights(model, filename):
#             sd = torch.load(filename, map_location=lambda storage, loc: storage)
#             names = set(model.state_dict().keys())
#             for n in list(sd.keys()): 
#                 if n not in names and n+'_raw' in names:
#                     if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
#                     del sd[n]
#             model.load_state_dict(sd)
            
#         #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         if face_alignment_net is None:
#             face_alignment_model = r"./models/2DFAN4-11f355bf06.pth.tar"    
#             #Face alignement
#             network_size = 4
#             face_alignment_net = FAN(network_size)
#             load_weights(face_alignment_net,face_alignment_model)
#             face_alignment_net.to(device)
#             face_alignment_net.eval()

#         if face_detection_net is None:
#             #face detection 
#             face_detector_model = r"./models/s3fd-619a316812.pth"
#             face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)
    
#     #localize the face in the video 
#     # localize_face = 0 -> Face is localized at a single frame in the video (the middle frame)
#     # localize_face = -1 -> Face is localized at each frame of the video
#     # localize_face = n -> face is localized every n frames 

#     # we will start by localizing the face in the middel of the video, if additional information is needed 
#     # then will be added as required

    
    
#     #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
#     video_fps = int(video_handler.get(cv2.CAP_PROP_FPS))
#     video_handler.set(cv2.CAP_PROP_POS_FRAMES, num_frames//2)

#     success, image = video_handler.read()
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     if success: 
#         detected_faces = face_detection_net.detect_from_image(image)
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#     video_handler.release()
    
#     # at this point we have the position of the face in the mid frame. Let's use that info
    
#     #extend the face bounding box to improve localization
#     if fix_head_position:
#         detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#         for i, d in enumerate(detected_faces):
#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale



#     #create a dataframe that will store all the information 
#     df_cols = ["Video_Frame_number", "bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0,68):
#         num=str(i)
#         xx = 'landmark_'+num+'_x'
#         yy = 'landmark_'+num+'_y'
#         df_cols.append(xx)
#         df_cols.append(yy)

#     LandmarkDataFrame = pd.DataFrame(columns = df_cols)

#     timestamps = []
#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     k = 0
#     success = True
#     while success:
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             if localize_face == 0:
#                 #do not localize the face, use previous info 
#                 pass 
#             elif localize_face == -1 :
#                 #localize the face at each frame, upd
#                 update_detected_faces = face_detection_net.detect_from_image(image)
#                 for i, d in enumerate(update_detected_face):

#                     if d[4]>=0.8:
#                         #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                         # if not (<0.8) don't update the bounding box
#                         detected_faces = update_detected_face
#                         center = torch.FloatTensor(
#                             [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                         center[1] = center[1] - (d[3] - d[1]) * 0.12
#                         scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
                        
#                         if fix_head_position:
#                             detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#                             for i, d in enumerate(detected_faces):
#                                 center = torch.FloatTensor(
#                                     [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                 center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                 scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#             else:
#                 #only update every n frames
#                 if (k+1)%localize_face == 0:
#                     update_detected_face = face_detection_net.detect_from_image(image)
#                     for i, d in enumerate(update_detected_face):

#                         if d[4]>=0.8:
#                             #do we trust the face localizer, if yes (>0.8) then update the bounding box, 
#                             # if not (<0.8) don't update the bounding box
#                             detected_faces = update_detected_face
#                             center = torch.FloatTensor(
#                                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                             center[1] = center[1] - (d[3] - d[1]) * 0.12
#                             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
                            
#                             if fix_head_position:
#                                 try: 
#                                     detected_faces[0][3] = detected_faces[0][3]+fix_head_position
#                                     for i, d in enumerate(detected_faces):
#                                         center = torch.FloatTensor(
#                                             [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#                                         center[1] = center[1] - (d[3] - d[1]) * 0.12
#                                         scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale
#                                 except:
#                                     pass


#             inp = utils.crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                         (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255).unsqueeze_(0)

#             out = face_alignment_net(inp)[-1].detach() #[-1] is to get the output of the last hourglass block
#             out = out.cpu()
#             pts, pts_img = utils.get_preds_fromhm(out, center, scale)

#             pts_img = pts_img.view(68, 2)



#             # Store everything in a dataframe
#             datus = []
#             datus.append(int(k)+1)  # frame number in the color_only video 

#             datus.append(detected_faces[0][0])  #top
#             datus.append(detected_faces[0][1])  #left
#             datus.append(detected_faces[0][2])  #bottom
#             datus.append(detected_faces[0][3])  #right

#             all_landmarks = pts_img.numpy()
#             for x,y in all_landmarks:
#                 datus.append(x), datus.append(y)  #x and y position of each landmark

#             LandmarkDataFrame = LandmarkDataFrame.append(pd.Series(datus,index = df_cols), 
#                                    ignore_index = True)
            
#             timestamps.append(video_handler.get(cv2.CAP_PROP_POS_MSEC)/1000)

#             k +=1 


#     #add time to landmarks 
#     LandmarkDataFrame.insert(loc=1, column='Time_Stamp (s)', value=timestamps)
#     base, file = os.path.split(color_file)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-4]+'_landmarks.csv')
#     LandmarkDataFrame.to_csv(landmark_file)   

#     return landmark_file

    


# def smooth_landmarks_video(landmark_file, create_video=False, sufix = 'landmarks'):
#     LandmarkDataFrame = pd.read_csv(landmark_file, index_col=0)
#     #b, a = signal.bessel(2 ,0.1)
#     windowlength=5
#     for i in range(68):
#         num=str(i)
#         xx = LandmarkDataFrame['landmark_'+num+'_x'].values
#         xx_med = signal.medfilt(xx,kernel_size=windowlength)
#     #     mod_xx = sm.tsa.statespace.SARIMAX(xx, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
#     #     res_xx = mod_xx.fit()
#     #     predict_xx = res_xx.get_prediction(end=mod_xx.nobs +0-1)
#     #     predict_xx_out = predict_xx.predicted_mean
#     #     predict_xx_out[0] = xx[0]


#         yy = LandmarkDataFrame['landmark_'+num+'_y'].values
#         yy_med = signal.medfilt(yy,kernel_size=windowlength)
#     #     mod_yy = sm.tsa.statespace.SARIMAX(yy, order=(ARdegree,0,MAdegree),seasonal_order=(0, 0, 0, 0),simple_differencing=True)
#     #     res_yy = mod_yy.fit()
#     #     predict_yy = res_yy.get_prediction(end=mod_yy.nobs +0-1)
#     #     predict_yy_out = predict_yy.predicted_mean
#     #     predict_yy_out[0] = yy[0]

#         LandmarkDataFrame['landmark_'+num+'_x'] = xx_med
#         LandmarkDataFrame['landmark_'+num+'_y'] = yy_med

        
#     landmark_file = landmark_file[:-4] + 'Filtered.csv'
#     LandmarkDataFrame.to_csv(landmark_file)

#     if create_video:

#         video_handler = cv2.VideoCapture(color_file)  # read the video
#         num_frames = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
#         FPS= int(video_handler.get(cv2.CAP_PROP_FPS))
#         width = int(video_handler.get(3)) # float
#         height = int(video_handler.get(4)) # float

#         color_file_landmark = landmark_file[:-4] + '.mp4'

#         video = VideoWriter(color_file_landmark, -1, int(FPS), (width,height))
#         video_handler = cv2.VideoCapture(color_file)  # read the video
#         success = True
#         k=0
#         for k in range(int(num_frames)):
#             success, image = video_handler.read()

#             frame_number=k+1
#             frame_information = LandmarkDataFrame.loc[LandmarkDataFrame['Video_Frame_number'] == frame_number].values
#             shape = np.array([frame_information[0][6:]])
#             shape = np.reshape(shape.astype(np.int), (-1, 2))
#             for (x, y) in shape:
#                 if x is np.NaN:
#                     continue
#                 else:
#                     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

#             frame_to_save = image
#             video.write(frame_to_save)
#             k +=1

#         video.release()

#         return landmark_file
        
        
# def estimate_3dlandmarks_video(cvs_file,
#                          device,
#                          color_file,
#                          sufix = 'landmarks',
#                          face_detection_net=None,
#                          depth_network=None):
#     import face_alignment.utils as utils

#     if  (depth_network is None) or (face_detection_net is None):

#         from face_alignment import api as face_alignment
#         from face_alignment.models import FAN, ResNetDepth
#         from face_alignment.detection.sfd import sfd_detector


#         if depth_network is None:
#             depth_network = ResNetDepth()
#             depth_model = r"./models/depth-2a464da4ea.pth.tar"
#             sd = torch.load(depth_model, map_location=lambda storage, loc: storage)
#             depth_dict = {
#                 k.replace('module.', ''): v for k,
#                                                 v in sd['state_dict'].items()}

#             depth_network.load_state_dict(depth_dict)
#             depth_network.to(device)
#             depth_network.eval()

#         if face_detection_net is None:
#             #face detection
#             face_detector_model = r"./models/s3fd-619a316812.pth"
#             face_detection_net = sfd_detector.SFDDetector(device=device, path_to_detector=face_detector_model, verbose=False)



#     DF_landmarks = pd.read_csv(cvs_file, index_col=0)

#     # create dataframe to store information about 3d position of landmarks
#     df_cols_p1 = ["Video_Frame_number", "bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
#     for i in range(0, 68):
#         num = str(i)
#         xx = 'landmark_' + num
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)
#         df_cols_p1.append(xx)

#     df_cols_p2 = ["", "", "", "", ""]
#     for i in range(0, 68):
#         df_cols_p2.append("x")
#         df_cols_p2.append("y")
#         df_cols_p2.append("z")

#     header = [np.array(df_cols_p1),
#               np.array(df_cols_p2)]

#     DF_3dpositions = pd.DataFrame(columns=header)

#     time_st = DF_landmarks['Time_Stamp (s)'].tolist()

#     # re-position the video handler at the first frame and start going frame by frame
#     video_handler = cv2.VideoCapture(color_file)  # read the video
#     k = 0

#     bbox_top_x = DF_landmarks['bbox_top_x'].tolist()
#     bbox_top_y = DF_landmarks['bbox_top_y'].tolist()
#     bbox_bottom_x = DF_landmarks['bbox_bottom_x'].tolist()
#     bbox_bottom_y = DF_landmarks['bbox_bottom_y'].tolist()

#     success = True
#     while success:
#         success, image = video_handler.read()
#         if success:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             d = torch.zeros(4)
#             d[0] = bbox_top_x[k]
#             d[1] = bbox_top_y[k]
#             d[2] = bbox_bottom_x[k]
#             d[3] = bbox_bottom_y[k]

#             center = torch.FloatTensor(
#                 [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
#             center[1] = center[1] - (d[3] - d[1]) * 0.12
#             scale = (d[2] - d[0] + d[3] - d[1]) / face_detection_net.reference_scale

#             inp = utils.crop(image, center, scale)
#             inp = torch.from_numpy(inp.transpose(
#                 (2, 0, 1))).float()
#             inp = inp.to(device)
#             inp.div_(255).unsqueeze_(0)

#             landmarks = DF_landmarks.loc[DF_landmarks['Video_Frame_number'] == k + 1].values

#             if len(landmarks) > 0:
#                 # continue only if landmarks for the frame are avaliable
#                 landmarks = landmarks[0][6:]
#                 landmarks = landmarks.astype('float').reshape(-1, 2)

#             heatmaps = torch.zeros((68, 256, 256))
#             for i in range(68):
#                 transformed_landmarks = transform(landmarks[i] + 1, center, scale, 256, invert=False)
#                 heatmaps[i] = draw_gaussian(heatmaps[i], transformed_landmarks, 2)

#             heatmaps = heatmaps.to(device).unsqueeze_(0)

#             depth_pred = depth_network(
#                 torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)

#             depth_pred_corrected = depth_pred * (1.0 / (256.0 / (200.0 * scale)))
#             depth_pred_corrected = depth_pred_corrected.numpy()

#             preds = np.column_stack((landmarks, depth_pred_corrected))

#             # Store everything in a dataframe
#             datus = []
#             datus.append(int(k) + 1)  # frame number in the color_only video

#             datus.append(d[0].item())  # top
#             datus.append(d[1].item())  # left
#             datus.append(d[2].item())  # bottom
#             datus.append(d[3].item())  # right

#             for x, y, z in preds:
#                 datus.append(x), datus.append(y), datus.append(z)  # x and y position of each landmark

#             DF_3dpositions = DF_3dpositions.append(pd.Series(datus, index=header),
#                                                    ignore_index=True)

#             k += 1

#     DF_3dpositions.insert(loc=1, column='Time_Stamp (s)', value=time_st)


#     base, file = os.path.split(color_file)
#     base = os.path.join(base,sufix)
#     landmark_file = os.path.join(base,file[:-4]+'_3DPredicted.csv')
#     DF_3dpositions.to_csv(landmark_file)
