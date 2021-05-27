
import pyrealsense2 as rs
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import VideoWriter, VideoWriter_fourcc

import torch
from scipy import signal, ndimage, spatial
import math











def get3dlandmarks(BAG_File, info_file, csv_file, sufix = 'landmarks'):

    DF_landmarks = pd.read_csv(csv_file, index_col=0)
    for col in DF_landmarks.columns:
        if 'z'in col:
            DF_landmarks = DF_landmarks.drop([col], axis = 1)


    DF_info = pd.read_csv(info_file, index_col=0)

    if len(DF_landmarks) != len(DF_info):
        print('Error')
        print("LEN OF CSV AND PIXEL VALUE DATA ARE NOT THE SAME")
        print()
        return


    # create dataframe to store information about 3d position of landmarks
    df_cols_p1 = ["Video_Frame_number", 'Time_Stamp (s)']
    for i in range(0, 68):
        num = str(i)
        xx = 'landmark_' + num
        df_cols_p1.append(xx + '_x')
        df_cols_p1.append(xx + '_y')
        df_cols_p1.append(xx + '_z')

    header = np.array(df_cols_p1)


    DF_3dpositions = pd.DataFrame(columns=header)

    DF_3dpositions['Time_Stamp (s)']= DF_info['Frame_Time_Stamp']
    DF_3dpositions['Video_Frame_number'] = DF_landmarks.Video_Frame_number.values

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

                index_ = DF_info.index[DF_info['BAG_Frame_Number']==this_frame].values[0]

                landmarks = DF_landmarks.iloc[index_].values[1:]

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

                for (c,r) in landmarks:  # landmarks provide the x,y position of each landmark. x are columns and y are rows in the figure
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
                continue


    except RuntimeError:
        pass
    finally:
        pipeline.stop()

    DF_3dpositions['depth_scale'] = depth_scale
    landmark_file = csv_file[:-4]+'3D.csv'
    DF_3dpositions.to_csv(landmark_file)
    return


def get3dlandmarks_video(filename_video, filename_csv):
    vid = cv2.VideoCapture(filename_video)

    landmarksDataFrame = pd.read_csv(filename_csv, index_col=0)
    for col in landmarksDataFrame.columns:
        if 'z'in col:
            landmarksDataFrame = landmarksDataFrame.drop([col], axis = 1)



    len_video = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    len_frames = len(landmarksDataFrame)

    h= int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w= int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    # create dataframe to store information about 3d position of landmarks
    df_cols_p1 = ["Video_Frame_number", 'Time_Stamp (s)']
    for i in range(0, 68):
        num = str(i)
        xx = 'landmark_' + num
        df_cols_p1.append(xx + '_x')
        df_cols_p1.append(xx + '_y')
        df_cols_p1.append(xx + '_z')

    header = np.array(df_cols_p1)
    DF_3dpositions = pd.DataFrame(columns=header)

    DF_3dpositions['Video_Frame_number'] = landmarksDataFrame.Video_Frame_number.values
    #DF_3dpositions[["Video_Frame_number", 'Time_Stamp (s)']]=landmarksDataFrame[["Video_Frame_number", 'Time_Stamp (s)']]

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


        landmarks_ = info_frame[1:].reshape(-1,2)

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
            frames = pipeline.wait_for_frames(100)  #get frame from file

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
