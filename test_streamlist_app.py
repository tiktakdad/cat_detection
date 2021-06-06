import streamlit as st
import numpy
import sys
import os
import tempfile

sys.path.append(os.getcwd())
#import traffic_counter as tc
import cv2
import time
#import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
import copy
#from components.custom_slider import custom_slider

# define the weights to be used along with its config file
config = 'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'

# define recommend values for model confidence and nms suppression
def_values = {'conf': 70, 'nms': 50}
keys = ['conf', 'nms']


@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)





def trigger_rerun():
    """
    mechanism in place to force resets and update widget states
    """
    session_infos = Server.get_current()._session_info_by_id.values()
    for session_info in session_infos:
        this_session = session_info.session
    this_session.request_rerun()


def main():
    st.set_page_config(page_title="Traffic Flow Counter",
                       page_icon=":vertical_traffic_light:")


    hide_streamlit_widgets()
    """
    #  Traffic Flow Counter :blue_car:  :red_car:
    Upload a video file to track and count vehicles. Don't forget to change parameters to tune the model!
    #### Features to be added in the future:
    + speed measurement
    + traffic density
    + vehicle type distribution
    """



    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()
    tfile = tempfile.NamedTemporaryFile(delete=True)

    upload.empty()
    vf = cv2.VideoCapture('/content/drive/MyDrive/mycat/video.mp4')
    ProcessFrames(vf, stop_button)




def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def ProcessFrames(vf,  stop):
    """
        main loop for processing video file:
        Params
        vf = VideoCapture Object
        tracker = Tracker Object that was instantiated
        obj_detector = Object detector (model and some properties)
    """

    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        print('Total number of frames to be processed:', num_frames,
              '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')

    frame_counter = 0
    _stop = stop.button("stop")
    new_car_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()

    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter / (end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter / num_frames)

        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frm, width=720)


main()