import streamlit as st
from PIL import Image
import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox
# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG

board_corners_facebow = [
    np.array(
        [
            [-0.0085, 0.026973, 0.011185],
            [0.0085, 0.026973, 0.011185],
            [0.0085, 0.013048, 0.001434],
            [-0.0085, 0.013048, 0.001434],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [-0.0085, 0.0085, 0],
            [0.0085, 0.0085, 0],
            [0.0085, -0.0085, 0],
            [-0.0085, -0.0085, 0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [-0.0085, -0.013048, 0.001434],
            [0.0085, -0.013048, 0.001434],
            [0.0085, -0.026974, 0.011185],
            [-0.0085, -0.026974, 0.011185],
        ],
        dtype=np.float32,
    ),

]

board_corners_upper = [
    np.array(
        [
            [-0.026064, -0.001102, 0.019],
            [-0.012138, -0.010853, 0.019],
            [-0.012138, -0.010853, 0.002],
            [-0.026064, -0.001102, 0.002],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [-0.0085, -0.012, 0.019],
            [0.0085, -0.012, 0.019],
            [0.0085, -0.012, 0.002],
            [-0.0085, -0.012, 0.002],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.012138, -0.010853, 0.019],
            [0.026064, -0.001102, 0.019],
            [0.026064, -0.001102, 0.002],
            [0.012138, -0.010853, 0.002],
        ],
        dtype=np.float32,
    ),
]

board_corners_lower = [
    np.array(
        [
            [-0.026064, -0.001102, -0.002],
            [-0.012138, -0.010853, -0.002],
            [-0.012138, -0.010853, -0.019],
            [-0.026064, -0.001102, -0.019],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [-0.0085, -0.012, -0.002],
            [0.0085, -0.012, -0.002],
            [0.0085, -0.012, -0.019],
            [-0.0085, -0.012, -0.019],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.012138, -0.010853, -0.002],
            [0.026064, -0.001102, -0.002],
            [0.026064, -0.001102, -0.019],
            [0.012138, -0.010853, -0.019],
        ],
        dtype=np.float32,
    ),
]
UpBoard_ids = np.array([[0], [1], [2]], dtype=np.int32)
LowBoard_ids = np.array([[3], [4], [5]], dtype=np.int32)
Right_TMJ_board_ids = np.array([[6], [7], [8]], dtype=np.int32)
Left_TMJ_board_ids = np.array([[9], [10], [11]], dtype=np.int32)
Nose_board_ids = np.array([[12], [13], [14]], dtype=np.int32)

UpBoard = aruco.Board_create(board_corners_upper, ARUCO_DICT, UpBoard_ids)
LowBoard = aruco.Board_create(board_corners_lower, ARUCO_DICT, LowBoard_ids)
Right_TMJ_board = aruco.Board_create(board_corners_facebow, ARUCO_DICT, Right_TMJ_board_ids)
Left_TMJ_board = aruco.Board_create(board_corners_facebow, ARUCO_DICT, Left_TMJ_board_ids)
Nose_board = aruco.Board_create(board_corners_facebow, ARUCO_DICT, Nose_board_ids)


def Calibrate(squareLength,markerLength,Cv2Images) :

    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
    CHARUCO_BOARD = aruco.CharucoBoard_create(5,7,squareLength,markerLength,ARUCO_DICT)   
    corners_all = []  # Corners discovered in all images processed
    ids_all = []  # Aruco ids corresponding to corners discovered
    counter = 0
    image_size = tuple()

    for Cv2img in Cv2Images:
        gray = cv2.cvtColor(Cv2img, cv2.COLOR_BGR2GRAY)
        if not image_size :
            image_size = gray.shape[::-1]
        else :
            if image_size != gray.shape[::-1] :
                res = 0
                message = [
                            "WARNING:",
                            "Calibration was unsuccessful!",
                            "Calibration Images have not uniforme size.",
                            "Please Remove Uploded Images, and try upload uniforme sized images,"
                            "from same camera and retry."  ]
                return res, message, None, None
            else :
                corners, ids, _ = aruco.detectMarkers(
                    image=gray, dictionary=ARUCO_DICT, parameters=ARUCO_PARAMETERS 
                )
                
                if corners :
                    response,charuco_corners,charuco_ids = aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=gray,
                        board=CHARUCO_BOARD,
                    )
                    if response > 20:
                        # Add these corners and ids to our calibration arrays
                        corners_all.append(charuco_corners)
                        ids_all.append(charuco_ids)
                        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                        counter+=1
        
    if counter < 10 :
        res = 0
        message = [
        "WARNING:",
        "Calibration was unsuccessful!",
        f"{counter}  processed Calibration Images.",
        "A minimum of 10 good images is required.",
        "Please try upload more images and retry."]
        return res, message, None, None

    if 10 <= counter < 20 :
        message = [
        f"{counter}  processed Calibration Images.",  
        "Precision : Calibration result may be imprecise.",
        "##################################################"]
    if 20 <= counter < 40 :
        message = [
        f"{counter} processed Calibration Images.",  
        "Precision : Good Calibration results.",
        "##################################################"]
    if 40 <= counter : 
        message = [
        f"{counter} processed Calibration Images.",  
        "Precision : Excellent Calibration results .",
        "##################################################"]

    calibration,cameraMatrix,distCoeffs,rvecs,tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        )
    res = 1
    return res, message, cameraMatrix, distCoeffs

###################################################################################################
def GetCamIntrisics_from_File(CalibFile):
    fn = CalibFile.name
    if fn.endswith('.pckl') :
        CalibFile.seek(0)
        (K, distCoeffs, _, _) = pickle.load(CalibFile)
    if fn.endswith('.txt') :
        
        with open(fn, 'r') as rf :
            linesRead = rf.readlines()
            K, distCoeffs = np.array( eval(linesRead[1]) ), np.array( eval(linesRead[3]) )
            
    fx, fy= K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return [(K, distCoeffs), (fx, fy, cx, cy)]

###################################################################################################

def bdental_facebow():    
    image = Image.open('images/logo.png')
    st.image(image, width=128)
    st.title('Easy angles')

def plot_3d(fx, fy, fz, lbx, lby, lbz):
    fig = plt.figure(figsize=(8,8))    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('$X$', fontsize=10)
    ax.set_ylabel('$Y$', fontsize=10)
    ax.set_zlabel('$Z$', fontsize=10)
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    ax.scatter(fx, fy, fz)
    ax.scatter(lbx, lby, lbz)
    return fig

all_images_classified = False
classified_image_list = []
def images_classifier(image_files, file_pckl, menu_jaws_positions, menu_image_key, menu_jaws_positions_user_list):
    menu_image_key = 0
    menu_jaws_positions_user_list = []
    global classified_image_list
    global all_images_classified
    for image_file in image_files:    
        if image_file and file_pckl and not all_images_classified:
            im = Image.open(image_file)
            st.image(im,use_column_width=True)
            option = st.radio('Select lower jaw position:', menu_jaws_positions, key = menu_image_key)
            menu_jaws_positions_user_list.append(option)
            menu_image_key = menu_image_key+1
            if (sorted(menu_jaws_positions_user_list)) == menu_jaws_positions:
                if st.button('Calculate angles'):
                    for i in range(5):
                        classified_image = menu_jaws_positions_user_list[i], image_files[i].name
                        classified_image_list.append(classified_image)
                    all_images_classified = True
                    return classified_image_list, all_images_classified
newrow = [0,0,0,1]
def get_facebow_matrix(T):
    tvec_x = [T[0][1], T[2][1]]
    tvec_y = [T[0][2], T[2][2]]
    tvec_z = [T[0][3], T[2][3]]
    tvec_x_sum, tvec_y_sum, tvec_z_sum = sum(tvec_x), sum(tvec_y), sum(tvec_z)
    Facebow_tvec = [[tvec_x_sum/2], [tvec_y_sum/2], [tvec_z_sum/2]]
    Facebow_tvec = np.asarray(Facebow_tvec)
    T1 = np.array([T[0][1], T[0][2], T[0][3]])
    T2 = np.array([T[1][1], T[1][2], T[1][3]])
    T3 = np.array([T[2][1], T[2][2], T[2][3]])
    X = T3 - T1
    Mid = .5 * (T1 + T3)
    Y = Mid - T2
    normX = np.linalg.norm(X)
    X = X/normX
    normY = np.linalg.norm(Y)
    Y = Y/normY
    Z = np.cross(X, Y)
    R = np.zeros((3, 3))
    R[0][0], R[0][1], R[0][2] = X[0], Y[0], Z[0]
    R[1][0], R[1][1], R[1][2] = X[1], Y[1], Z[1]
    R[2][0], R[2][1], R[2][2] = X[2], Y[2], Z[2]
    Facebow_matrix = np.concatenate((R, Facebow_tvec), axis=1)
    Facebow_matrix = np.vstack([Facebow_matrix, newrow])
    
    return Facebow_matrix
    
def getAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                text="", textposition="outside", text_kw=None, **kwargs):
        
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                        theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                    xycoords=IdentityTransform(),
                    xytext=(0, 0), textcoords="offset points",
                    annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                        [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                    (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])

MarkersIdCornersDict = dict()
objects_poses = {}
def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('BDENTAL AXIOGRAPH', 'CAMERA CALIBRATION TOOL')
    )

    #########################################################################################
    ########################### CAMERA CALIBRATION TOOL Page ################################
    #########################################################################################
    if selected_box == 'CAMERA CALIBRATION TOOL':
        ############# Add some User Info ######################################################
        INFO =  [
            'BDENTAL Calibration tool is based on Opencv Charuco board detection.',
            'It is a robust camera calibration method, that will generate a .txt file,',
            'containing Camera Calibration parameters.',
            'For consistant and precise results please refere to this recommendations :',
            '_Minimum_ : 10 photos, _Optimal_ : 20 photos, _Good_ : 30 photos, _Excellent_ : 40+ photos',
            'You can download the charuco board from :',
            ]
        for line in INFO :
            st.text(line)
        st.write("[Charuco Board pdf download link.](https://github.com/issamdakir/BDENTAL4D-LINUX/blob/291/Resources/ForPrinting/CalibrationBoardA4.pdf)")
                
        st.subheader('Upload camera calibration photos :')
        ###########################################################################################
        CalibFiles = st.file_uploader("", type=['png', 'jpg','tif','tiff','TIF','TIFF', 'jpeg','bmp','webp','pfm','sr','ras','exr','hdr','pic'], accept_multiple_files=True)
        
        if CalibFiles :
            CaseLength = st.number_input(label='Square width', min_value=0.00, max_value=None, value=24.40, help='The width in mm of the calibration board Square')                
            MarkerLength = st.number_input(label='Marker width', min_value=0.00, max_value=None, value=12.30, help='The width in mm of the calibration board Marker')                
            CalibrateButton = st.button('CALIBRATE CAMERA')
            if CalibrateButton :
                Processing = st.empty()
                Processing.text('Please wait while processing, results will be displayed within few secondes...')
                Cv2Images = []
                for f in CalibFiles:
                    try :
                        img = Image.open(f)
                        Cv2img = np.array(img)
                        if Cv2img.size > 1 :
                            Cv2Images.append(Cv2img)
                    except Exception as Error:
                        print(f'cant open {f.name}')
                        print(Error)
                        continue
                res, message, cameraMatrix, distCoeffs = Calibrate(CaseLength,MarkerLength,Cv2Images)
                Processing.text('')
                st.subheader('*Camera calibration results :*')
                for line in message :
                    st.text(line)
                if res :
                    st.write('K Matrix : ',cameraMatrix)
                    st.write('Distorsion coefficients : ',distCoeffs)
                    cameraMatrix = str(cameraMatrix.tolist())
                    distCoeffs = str(distCoeffs.tolist())
                    data_calibration = '''Camera Matrix :\n'''+cameraMatrix+'''\nDistortion Coefficients :\n'''+distCoeffs
                    st.download_button(
                        label="Save Calibration file",
                        data=data_calibration,
                        file_name='calibration.txt',
                        mime='text/csv',
                    )
    
    if selected_box == 'BDENTAL AXIOGRAPH':
        bdental_facebow()
        #st.subheader('Load Image(s)')
        st.markdown('### Load photos')
        image_files = st.file_uploader("", type=['png', 'jpg'], accept_multiple_files=True)
        #st.subheader('Load Calibration file')
        st.markdown('### Load Calib file')
        CalibFile = st.file_uploader("", type=([".pckl", "txt"]))
        menu_jaws_positions = ['Central Relation',
            'Laterotrusion Left',
            'Laterotrusion Right',
            'Maximum Opening',
            'Protrusion'
        ]
        menu_image_key = 0
        menu_jaws_positions_user_list = []
        images_classifier(image_files, CalibFile, menu_jaws_positions, menu_image_key, menu_jaws_positions_user_list)
        if all_images_classified:
            (cameraMatrix, distCoeffs), _ = GetCamIntrisics_from_File(CalibFile)
            for image_file in image_files:
                im = Image.open(image_file)
                im = np.array(im)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if cameraMatrix is None or distCoeffs is None:
                    st.text("Calibration issue")
                else:
                    corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS, cameraMatrix=cameraMatrix, distCoeff=distCoeffs)
                    if np.all(ids is not None):
                        facebow_tvecs = []
                        for i in range(len(ids)):
                            MarkersIdCornersDict[ids[i][0]] = (list(corners))[i]
                            pts = MarkersIdCornersDict[ids[i][0]].astype(int)
                            pts = pts.reshape((-1,1,2))
                            cv2.fillConvexPoly(	im, pts, (255,174,0))
                        FaceBow_ids_list = np.array([6,7,8,9,10,11,12,13,14])
                        condition_facebow = FaceBow_ids_list in ids
                        condition_upboard = UpBoard_ids[0] in ids
                        condition_lowboard = LowBoard_ids[0] in ids
                        ####################################################
                        #Facebow Boards
                        ####################################################
                        #Right TMJ Board
                        if condition_facebow:
                            Right_TMJ_Corners = [
                                MarkersIdCornersDict[6],
                                MarkersIdCornersDict[7],
                                MarkersIdCornersDict[8],
                            ]
                            Right_TMJ_retval, Right_TMJ_Rvec, Right_TMJ_Tvec = cv2.aruco.estimatePoseBoard(
                                Right_TMJ_Corners,
                                Right_TMJ_board_ids,
                                Right_TMJ_board,
                                cameraMatrix,
                                distCoeffs,
                                None,
                                None,
                            )
                            #aruco.drawAxis(im, cameraMatrix, distCoeffs, Right_TMJ_Rvec, Right_TMJ_Tvec, 0.03)
                            Right_TMJ_Tvec = Right_TMJ_Tvec.reshape((3, 1))*1000
                            Right_TMJ_Tvec = ['6', Right_TMJ_Tvec[0][0], Right_TMJ_Tvec[2][0], -Right_TMJ_Tvec[1][0]]
                            facebow_tvecs.append(Right_TMJ_Tvec)
                        #Left TMJ Board
                            Left_TMJ_Corners = [
                                MarkersIdCornersDict[9],
                                MarkersIdCornersDict[10],
                                MarkersIdCornersDict[11],
                            ]
                            Left_TMJ_retval, Left_TMJ_Rvec, Left_TMJ_Tvec = cv2.aruco.estimatePoseBoard(
                                Left_TMJ_Corners,
                                Left_TMJ_board_ids,
                                Left_TMJ_board,
                                cameraMatrix,
                                distCoeffs,
                                None,
                                None,
                            )
                            Left_TMJ_Tvec = Left_TMJ_Tvec.reshape((3, 1))*1000
                            Left_TMJ_Tvec = ['8', Left_TMJ_Tvec[0][0], Left_TMJ_Tvec[2][0], -Left_TMJ_Tvec[1][0]]
                            facebow_tvecs.append(Left_TMJ_Tvec)
                        #Nose Board
                            Nose_Corners = [
                                MarkersIdCornersDict[12],
                                MarkersIdCornersDict[13],
                                MarkersIdCornersDict[14],
                            ]
                            Nose_retval, Nose_Rvec, Nose_Tvec = cv2.aruco.estimatePoseBoard(
                                Nose_Corners,
                                Nose_board_ids,
                                Nose_board,
                                cameraMatrix,
                                distCoeffs,
                                None,
                                None,
                            )
                            Nose_Tvec = Nose_Tvec.reshape((3, 1))*1000
                            Nose_Tvec = ['7', Nose_Tvec[0][0], Nose_Tvec[2][0], -Nose_Tvec[1][0]]
                            facebow_tvecs.append(Nose_Tvec)
                        ####################################################
                        #UpBoard
                        ####################################################
                        if condition_upboard: 
                            UpCorners = [
                                MarkersIdCornersDict[0],
                                MarkersIdCornersDict[1],
                                MarkersIdCornersDict[2],
                            ]
                            Upretval, Up_Rvec, Up_Tvec = cv2.aruco.estimatePoseBoard(
                                UpCorners,
                                UpBoard_ids,
                                UpBoard,
                                cameraMatrix,
                                distCoeffs,
                                None,
                                None,
                            )
                            Up_Rvec, _ = cv2.Rodrigues(Up_Rvec)
                            Up_Tvec = Up_Tvec.reshape((3, 1))
                            Up_Board_matrix = np.concatenate((Up_Rvec, Up_Tvec*1000), axis=1)
                            Up_Board_matrix[[1, 2]] = Up_Board_matrix[[2, 1]]
                            c = np.array([[1,1,1,1], [1,1,1,1], [-1,-1,-1,-1]])
                            Up_Board_matrix = Up_Board_matrix*c
                            Up_Board_matrix = np.vstack([Up_Board_matrix, newrow])
                            Up_Board_matrix = Up_Board_matrix.tolist()
                        ####################################################
                        #LowBoard
                        ####################################################
                        if condition_lowboard: 
                            LowCorners = [
                                MarkersIdCornersDict[3],
                                MarkersIdCornersDict[4],
                                MarkersIdCornersDict[5],
                            ]
                            Lowretval, Low_Rvec, Low_Tvec = cv2.aruco.estimatePoseBoard(
                                LowCorners,
                                LowBoard_ids,
                                LowBoard,
                                cameraMatrix,
                                distCoeffs,
                                None,
                                None,
                            )
                            Low_Rvec, _ = cv2.Rodrigues(Low_Rvec)
                            Low_Tvec = Low_Tvec.reshape((3, 1))
                            Low_Board_matrix = np.concatenate((Low_Rvec, Low_Tvec*1000), axis=1)
                            Low_Board_matrix[[1, 2]] = Low_Board_matrix[[2, 1]]
                            c = np.array([[1,1,1,1], [1,1,1,1], [-1,-1,-1,-1]])
                            Low_Board_matrix = Low_Board_matrix*c
                            Low_Board_matrix = np.vstack([Low_Board_matrix, newrow])
                            Low_Board_matrix = Low_Board_matrix.tolist()
                        #cv2.aruco.drawDetectedMarkers(im,corners,ids)
                    #######################################
                    #Write data to dict
                    #######################################
                    objects_poses[image_file.name] = {
                        "Facebow_tvecs" : facebow_tvecs,
                        "Up_Board_matrix" : Up_Board_matrix,
                        "Low_Board_matrix" : Low_Board_matrix
                    }
                    st.image(im,use_column_width=True)
            ##########################################
            UpBoard_Matrix = np.zeros((4, 4))
            Facebow_zero_matrix = np.eye(4)
            left_cond = np.eye(4)
            left_cond[0][3], left_cond[1][3] = 55, -12
            right_cond = np.eye(4)
            right_cond[0][3], right_cond[1][3] = -55, -12
            left_cond_local, right_cond_local = np.eye(4), np.eye(4)
            # shift_all = np.zeros((4, 4))
            # shift_all[1][3] = 12
            left_cond_list = []
            right_cond_list = []
            incisial_list =[]
            left_cond_list.append((left_cond[0][3], left_cond[1][3], left_cond[2][3]))
            right_cond_list.append((right_cond[0][3], right_cond[1][3], right_cond[2][3]))
            #UpBoard_local_CR = np.zeros((4, 4))
            #UpBoard_transpose_CR = np.zeros((4, 4))
            Facebow_matrix_CR = np.zeros((4, 4))
            for classified_image in sorted(classified_image_list):
                #st.markdown('**'+str(classified_image[0])+'**')
                #############################################
                #STABILIZATION
                #############################################
                if classified_image[0] == "Central Relation":
                    Facebow_matrix_CR = np.array(get_facebow_matrix(sorted(objects_poses[classified_image[1]]['Facebow_tvecs'])))
                    UpBoard_world_matrix_CR = np.array(objects_poses[classified_image[1]]['Up_Board_matrix'])
                    LowBoard_world_matrix_CR = np.array(objects_poses[classified_image[1]]['Low_Board_matrix'])
                    left_cond_world = Facebow_matrix_CR @ left_cond
                    left_cond_local = np.linalg.inv(LowBoard_world_matrix_CR) @ left_cond_world 
                    right_cond_world = Facebow_matrix_CR @ right_cond
                    right_cond_local = np.linalg.inv(LowBoard_world_matrix_CR) @ right_cond_world 
                    Low_Board_local = np.linalg.inv(Facebow_matrix_CR) @ LowBoard_world_matrix_CR
                    Low_Board_local_transposed = Facebow_zero_matrix @ Low_Board_local
                    incisial_list.append((Low_Board_local_transposed[0][3], Low_Board_local_transposed[1][3], Low_Board_local_transposed[2][3]))
                if not classified_image[0] == "Central Relation":
                    UpBoard_world_matrix = np.array(objects_poses[classified_image[1]]['Up_Board_matrix'])
                    LowBoard_world_matrix = np.array(objects_poses[classified_image[1]]['Low_Board_matrix'])
                    LowBoard_local = np.linalg.inv(UpBoard_world_matrix) @ LowBoard_world_matrix

                    LowBoard_transposed = UpBoard_world_matrix_CR @ LowBoard_local
                    left_cond_transposed = LowBoard_transposed @ left_cond_local
                    left_cond_local_2 = np.linalg.inv(Facebow_matrix_CR) @ left_cond_transposed
                    left_cond_local_transposed = Facebow_zero_matrix @ left_cond_local_2

                    right_cond_transposed = LowBoard_transposed @ right_cond_local
                    right_cond_local_2 = np.linalg.inv(Facebow_matrix_CR) @ right_cond_transposed
                    right_cond_local_transposed = Facebow_zero_matrix @ right_cond_local_2

                    LowBoard_local_2 = np.linalg.inv(Facebow_matrix_CR) @ LowBoard_transposed
                    Low_Board_local_transposed = Facebow_zero_matrix @ LowBoard_local_2
                    
                    incisial_list.append((Low_Board_local_transposed[0][3], Low_Board_local_transposed[1][3], Low_Board_local_transposed[2][3]))
                    left_cond_list.append((left_cond_local_transposed[0][3], left_cond_local_transposed[1][3], left_cond_local_transposed[2][3]))
                    right_cond_list.append((right_cond_local_transposed[0][3], right_cond_local_transposed[1][3], right_cond_local_transposed[2][3]))
            #############################################
            #POINTS AND LINES
            #############################################
            st.markdown('**RESULTS**')

            #INCISIAL POINTS COORDS######################
            IP_CR_X, IP_CR_Y, IP_CR_Z = incisial_list[0][0], incisial_list[0][1], incisial_list[0][2]
            IP_LL_X, IP_LL_Y, IP_LL_Z = incisial_list[1][0], incisial_list[1][1], incisial_list[1][2]
            IP_LR_X, IP_LR_Y, IP_LR_Z = incisial_list[2][0], incisial_list[2][1], incisial_list[2][2]
            IP_MO_X, IP_MO_Y, IP_MO_Z = incisial_list[3][0], incisial_list[3][1], incisial_list[3][2]
            IP_PR_X, IP_PR_Y, IP_PR_Z = incisial_list[4][0], incisial_list[4][1], incisial_list[4][2]
            #LEFT COND POINTS COORDS#####################
            LC_CR_X, LC_CR_Y, LC_CR_Z = left_cond_list[0][0], left_cond_list[0][1], left_cond_list[0][2]
            LC_LL_X, LC_LL_Y, LC_LL_Z = left_cond_list[1][0], left_cond_list[1][1], left_cond_list[1][2]
            LC_LR_X, LC_LR_Y, LC_LR_Z = left_cond_list[2][0], left_cond_list[2][1], left_cond_list[2][2]
            LC_MO_X, LC_MO_Y, LC_MO_Z = left_cond_list[3][0], left_cond_list[3][1], left_cond_list[3][2]
            LC_PR_X, LC_PR_Y, LC_PR_Z = left_cond_list[4][0], left_cond_list[4][1], left_cond_list[4][2]
            #RIGHT COND POINTS COORDS####################
            RC_CR_X, RC_CR_Y, RC_CR_Z = right_cond_list[0][0], right_cond_list[0][1], right_cond_list[0][2]
            RC_LL_X, RC_LL_Y, RC_LL_Z = right_cond_list[1][0], right_cond_list[1][1], right_cond_list[1][2]
            RC_LR_X, RC_LR_Y, RC_LR_Z = right_cond_list[2][0], right_cond_list[2][1], right_cond_list[2][2]
            RC_MO_X, RC_MO_Y, RC_MO_Z = right_cond_list[3][0], right_cond_list[3][1], right_cond_list[3][2]
            RC_PR_X, RC_PR_Y, RC_PR_Z = right_cond_list[4][0], right_cond_list[4][1], right_cond_list[4][2]
            #############################################
            #Zero position INCISIAL POINT
            IP_LL_X, IP_LL_Y, IP_LL_Z = IP_LL_X - IP_CR_X, IP_LL_Y - IP_CR_Y, IP_LL_Z - IP_CR_Z
            IP_LR_X, IP_LR_Y, IP_LR_Z = IP_LR_X - IP_CR_X, IP_LR_Y - IP_CR_Y, IP_LR_Z - IP_CR_Z
            IP_MO_X, IP_MO_Y, IP_MO_Z = IP_MO_X - IP_CR_X, IP_MO_Y - IP_CR_Y, IP_MO_Z - IP_CR_Z
            IP_PR_X, IP_PR_Y, IP_PR_Z = IP_PR_X - IP_CR_X, IP_PR_Y - IP_CR_Y, IP_PR_Z - IP_CR_Z

            #Zero position LEFT COND
            LC_LL_X, LC_LL_Y, LC_LL_Z = LC_LL_X - LC_CR_X, LC_LL_Y - LC_CR_Y, LC_LL_Z - LC_CR_Z
            LC_LR_X, LC_LR_Y, LC_LR_Z = LC_LR_X - LC_CR_X, LC_LR_Y - LC_CR_Y, LC_LR_Z - LC_CR_Z
            LC_MO_X, LC_MO_Y, LC_MO_Z = LC_MO_X - LC_CR_X, LC_MO_Y - LC_CR_Y, LC_MO_Z - LC_CR_Z
            LC_PR_X, LC_PR_Y, LC_PR_Z = LC_PR_X - LC_CR_X, LC_PR_Y - LC_CR_Y, LC_PR_Z - LC_CR_Z
            #############################################
            #Zero position RIGHT COND
            RC_LL_X, RC_LL_Y, RC_LL_Z = RC_LL_X - RC_CR_X, RC_LL_Y - RC_CR_Y, RC_LL_Z - RC_CR_Z
            RC_LR_X, RC_LR_Y, RC_LR_Z = RC_LR_X - RC_CR_X, RC_LR_Y - RC_CR_Y, RC_LR_Z - RC_CR_Z
            RC_MO_X, RC_MO_Y, RC_MO_Z = RC_MO_X - RC_CR_X, RC_MO_Y - RC_CR_Y, RC_MO_Z - RC_CR_Z
            RC_PR_X, RC_PR_Y, RC_PR_Z = RC_PR_X - RC_CR_X, RC_PR_Y - RC_CR_Y, RC_PR_Z - RC_CR_Z
            #####
            LC_CR_X, LC_CR_Y, LC_CR_Z = 0, 0, 0
            RC_CR_X, RC_CR_Y, RC_CR_Z = 0, 0, 0
            IP_CR_X, IP_CR_Y, IP_CR_Z = 0, 0, 0
            
            #LINES INCISIAL POINT
            LineOpenX = [IP_CR_X, IP_MO_X]
            LineOpenY = [IP_CR_Y, IP_MO_Y]
            LineOpenZ = [IP_CR_Z, IP_MO_Z]
            
            LineProtrusionX = [IP_CR_X, IP_PR_X]
            LineProtrusionY = [IP_CR_Y, IP_PR_Y]
            LineProtrusionZ = [IP_CR_Z, IP_PR_Z]
            
            LineLatRightX = [IP_CR_X, IP_LR_X]
            LineLatRightY = [IP_CR_Y, IP_LR_Y]
            LineLatRightZ = [IP_CR_Z, IP_LR_Z]
            
            LineLatLeftX = [IP_CR_X, IP_LL_X]
            LineLatLeftY = [IP_CR_Y, IP_LL_Y]
            LineLatLeftZ = [IP_CR_Z, IP_LL_Z]
            
            LineMaxOpenY = [IP_MO_Y, IP_MO_Y]
            LineMaxOpenZ = [IP_CR_Z, IP_MO_Z]
            #LINES LEFT COND##############################
            LC_LineOpenX = [LC_CR_X, LC_MO_X]
            LC_LineOpenY = [LC_CR_Y, LC_MO_Y]
            LC_LineOpenZ = [LC_CR_Z, LC_MO_Z]

            LC_LineProtrusionX = [LC_CR_X, LC_PR_X]
            LC_LineProtrusionY = [LC_CR_Y, LC_PR_Y]
            LC_LineProtrusionZ = [LC_CR_Z, LC_PR_Z]

            LC_LineLatRightX = [LC_CR_X, LC_LR_X]
            LC_LineLatRightY = [LC_CR_Y, LC_LR_Y]
            LC_LineLatRightZ = [LC_CR_Z, LC_LR_Z]

            LC_LineLatLeftX = [LC_CR_X, LC_LL_X]
            LC_LineLatLeftY = [LC_CR_Y, LC_LL_Y]
            LC_LineLatLeftZ = [LC_CR_Z, LC_LL_Z]

            LC_LineMomShiftX = [LC_CR_X, LC_LL_X]
            LC_LineMomShiftY = [LC_CR_Y, LC_CR_Y]
            #LINES RIGHT COND##############################
            RC_LineOpenX = [RC_CR_X, RC_MO_X]
            RC_LineOpenY = [RC_CR_Y, RC_MO_Y]
            RC_LineOpenZ = [RC_CR_Z, RC_MO_Z]

            RC_LineProtrusionX = [RC_CR_X, RC_PR_X]
            RC_LineProtrusionY = [RC_CR_Y, RC_PR_Y]
            RC_LineProtrusionZ = [RC_CR_Z, RC_PR_Z]

            RC_LineLatRightX = [RC_CR_X, RC_LR_X]
            RC_LineLatRightY = [RC_CR_Y, RC_LR_Y]
            RC_LineLatRightZ = [RC_CR_Z, RC_LR_Z]

            RC_LineLatLeftX = [RC_CR_X, RC_LL_X]
            RC_LineLatLeftY = [RC_CR_Y, RC_LL_Y]
            RC_LineLatLeftZ = [RC_CR_Z, RC_LL_Z]

            RC_LineMomShiftX = [RC_CR_X, RC_LR_X]
            RC_LineMomShiftY = [RC_CR_Y, RC_CR_Y]
            #############################################
            #PLOTTING
            #############################################
            fig = plt.figure(figsize=(11.69, 18))
            fig.patch.set_facecolor("silver")
            spec = gridspec.GridSpec(
                ncols=4,
                nrows=6,
                figure=fig,
                left=0.07,
                bottom=0.13,
                right=0.95,
                top=0.97,
                wspace=0.9,
                hspace=1.5,
            )
            infospec = gridspec.GridSpec(
                ncols=4,
                nrows=12,
                figure=fig,
                left=0.07,
                bottom=0.1,
                right=0.95,
                top=1,
                wspace=0.9,
                hspace=1.5,
            )
            headimgalpha = 0.5

            # FRONTAL PLANE###########################################################################
            ax0 = fig.add_subplot(spec[0:2, 0:2])
            ax0.patch.set_facecolor("whitesmoke")
            ax0.set_title("Coronal Plane (incisal point)")
            ax0.set(xlabel="X axis, mm", ylabel="Z axis, mm")
            ax0.axis('equal')
            ax0.xaxis.set_major_locator(MultipleLocator(5))
            ax0.yaxis.set_major_locator(MultipleLocator(5))
            ax0.xaxis.set_minor_locator(MultipleLocator(1))
            ax0.yaxis.set_minor_locator(MultipleLocator(1))
            ax0.grid(which="minor", color="#e4e4e4", linestyle="--")

            ax0.grid(which="major", color="#CCCCCC", linestyle="--")
            ax0.grid(True)
            ax0.axhline(y=IP_CR_Z, color="black", linestyle="--", linewidth=2)
            ax0.axvline(x=IP_CR_X, color="black", linestyle="--", linewidth=2)
            ax0.plot(
                LineOpenX,
                LineOpenZ,
                color="red",
                linewidth=2,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
                label="Open",
            )
            ax0.plot(
                LineProtrusionX,
                LineProtrusionZ,
                color="blue",
                linewidth=2,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
                label="Protrusion",
            )
            ax0.plot(
                LineLatRightX,
                LineLatRightZ,
                color="orange",
                linewidth=2,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
                label="Laterotrusion Right",
            )
            ax0.plot(
                LineLatLeftX,
                LineLatLeftZ,
                color="magenta",
                linewidth=2,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
                label="Laterotrusion Left",
            )

            plt.legend(markerscale=0)

            ax = fig.add_subplot(spec[1, 1])
            image = plt.imread("images/incfront.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")

            # ANGLES
            # RIGHT LATEROTRUSION
            center = (IP_CR_X, IP_CR_Z)
            p1 = np.array([(IP_CR_X, IP_CR_Z), (IP_LR_X, IP_CR_Z)])
            p2 = np.array([(IP_CR_X, IP_CR_Z), (IP_LR_X, IP_LR_Z)])
            
            a=np.array([IP_LR_X, IP_CR_Z])
            b=np.array([IP_CR_X, IP_CR_Z])
            c=np.array([IP_LR_X, IP_LR_Z])
            ANGLE_LR = round(np.degrees(getAngle(a, b, c)), 2)
            if IP_LR_Z>IP_CR_Z:
                am0 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=100, text=ANGLE_LR, linewidth=3, zorder=1)
            else:
                am0 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=100, text=ANGLE_LR, linewidth=3, zorder=1)

            # LEFT LATEROTRUSION
            center = (IP_CR_X, IP_CR_Z)
            p1 = np.array([(IP_CR_X, IP_CR_Z), (IP_LL_X, IP_CR_Z)])
            p2 = np.array([(IP_CR_X, IP_CR_Z), (IP_LL_X, IP_LL_Z)])

            a=np.array([IP_LL_X, IP_CR_Z])
            b=np.array([IP_CR_X, IP_CR_Z])
            c=np.array([IP_LL_X, IP_LL_Z])
            ANGLE_LL = round(np.degrees(getAngle(a, b, c)), 2)
            
            if IP_LL_Z>IP_CR_Z:
                am1 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=100, text=ANGLE_LL, linewidth=3, zorder=1)
            else:
                am1 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=100, text=ANGLE_LL, linewidth=3, zorder=1)

            # OPENING DEVIATION
            center = (IP_CR_X, IP_CR_Z)
            p1 = np.array([(IP_CR_X, IP_CR_Z), (IP_MO_X, IP_MO_Z)])
            p2 = np.array([(IP_CR_X, IP_CR_Z), (IP_CR_X, IP_MO_Z)])

            a=np.array([IP_MO_X, IP_MO_Z])
            b=np.array([IP_CR_X, IP_CR_Z])
            c=np.array([IP_CR_X, IP_MO_Z])
            ANGLE_OD = round(np.degrees(getAngle(a, b, c)), 2)
            if IP_MO_X<IP_CR_X:
                am3 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=275, text=ANGLE_OD, linewidth=3, zorder=1)
                OD_side=str('right')
            else:
                am3 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=275, text=ANGLE_OD, linewidth=3, zorder=1)
                OD_side=str('left')
            # SAGITTAL PLANE############################################################################
            ax1 = fig.add_subplot(spec[0:2, 2:4])
            ax1.patch.set_facecolor("whitesmoke")
            plt.gca().invert_xaxis()
            ax1.axis('equal')
            ax1.set_title("Sagittal Plane (incisal point)")
            ax1.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
            ax1.xaxis.set_major_locator(MultipleLocator(5))
            ax1.yaxis.set_major_locator(MultipleLocator(5))
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
            ax1.yaxis.set_minor_locator(MultipleLocator(1))
            ax1.grid(which="minor", color="#e4e4e4", linestyle="--")
            ax1.grid(which="major", color="#CCCCCC", linestyle="--")
            ax1.grid(True)
            ax1.axhline(y=IP_CR_Z, color="black", linestyle="--", linewidth=2)
            ax1.axvline(x=IP_CR_Y, color="black", linestyle="--", linewidth=2)
            ax1.plot(
                LineMaxOpenY,
                LineMaxOpenZ,
                color="b",
                linestyle="--",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=2,
            )
            ax1.plot(
                LineOpenY,
                LineOpenZ,
                color="red",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax1.plot(
                LineProtrusionY,
                LineProtrusionZ,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax1.plot(
                LineLatRightY,
                LineLatRightZ,
                color="orange",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax1.plot(
                LineLatLeftY,
                LineLatLeftZ,
                color="magenta",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )

            ax = fig.add_subplot(spec[1, 3])
            image = plt.imread("images/incright.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")

            # ANGLES

            # PROTRUSION
            center = (IP_CR_Y, IP_CR_Z)
            p1 = np.array([(IP_CR_Y, IP_CR_Z), (IP_PR_Y, IP_PR_Z)])
            p2 = np.array([(IP_CR_Y, IP_CR_Z), (IP_PR_Y, IP_CR_Z)])

            a=np.array([IP_PR_Y, IP_PR_Z])
            b=np.array([IP_CR_Y, IP_CR_Z])
            c=np.array([IP_PR_Y, IP_CR_Z])
            ANGLE_Rrot = round(np.degrees(getAngle(a, b, c)), 2)
            if IP_PR_Z<IP_CR_Z:
                am4 = AngleAnnotation(center, p1[1], p2[1], ax=ax1, size=100, text=ANGLE_Rrot, linewidth=3, zorder=1)
            else:
                am4 = AngleAnnotation(center, p2[1], p1[1], ax=ax1, size=100, text=ANGLE_Rrot, linewidth=3, zorder=1)



            # INFO PLOTS########################################################################

            MaxOpen = abs(round((IP_MO_Z) - (IP_CR_Z), 2))

            axI1 = fig.add_subplot(infospec[4, 0])
            axI1.grid(False)
            plt.axis("off")
            axI1.text(
                0.0,
                1.2,
                "Laterotrusion Right = "
                + str(ANGLE_LR)
                + "°"
                + "\n"
                + "Laterotrusion Left = "
                + str(ANGLE_LL)
                + "°"
                + "\n"
                + "Opening deviation = "
                + OD_side
                + " "
                + str(ANGLE_OD)
                + "°",
            )

            axI2 = fig.add_subplot(infospec[4, 2])
            axI2.grid(False)
            plt.axis("off")
            axI2.text(
                0.0,
                1.2,
                "Incisal point protrusion = "
                + str(ANGLE_Rrot)
                + "°"
                + "\n"
                + "Maximum opening = "
                + str(MaxOpen)
                + " mm"
                + "\n",
            )

            ###################################################################################
            # CONDYLES LINES
            ###################################################################################
            RC_CP_X, RC_CP_Y, RC_CP_Z = 0, 0, 0
            LC_CP_X, LC_CP_Y, LC_CP_Z = 0, 0, 0

            # LEFT CONDYLE
            LC_LineOpenX = [LC_CP_X, LC_MO_X]
            LC_LineOpenY = [LC_CP_Y, LC_MO_Y]
            LC_LineOpenZ = [LC_CP_Z, LC_MO_Z]

            LC_LineProtrusionX = [LC_CP_X, LC_PR_X]
            LC_LineProtrusionY = [LC_CP_Y, LC_PR_Y]
            LC_LineProtrusionZ = [LC_CP_Z, LC_PR_Z]

            LC_LineLatRightX = [LC_CP_X, LC_LR_X]
            LC_LineLatRightY = [LC_CP_Y, LC_LR_Y]
            LC_LineLatRightZ = [LC_CP_Z, LC_LR_Z]

            LC_LineLatLeftX = [LC_CP_X, LC_LL_X]
            LC_LineLatLeftY = [LC_CP_Y, LC_LL_Y]
            LC_LineLatLeftZ = [LC_CP_Z, LC_LL_Z]

            LC_LineMomShiftX = [LC_CP_X, LC_LL_X]
            LC_LineMomShiftY = [LC_CP_Y, LC_CP_Y]

            # RIGHT CONDYLE
            RC_LineOpenX = [RC_CP_X, RC_MO_X]
            RC_LineOpenY = [RC_CP_Y, RC_MO_Y]
            RC_LineOpenZ = [RC_CP_Z, RC_MO_Z]

            RC_LineProtrusionX = [RC_CP_X, RC_PR_X]
            RC_LineProtrusionY = [RC_CP_Y, RC_PR_Y]
            RC_LineProtrusionZ = [RC_CP_Z, RC_PR_Z]

            RC_LineLatRightX = [RC_CP_X, RC_LR_X]
            RC_LineLatRightY = [RC_CP_Y, RC_LR_Y]
            RC_LineLatRightZ = [RC_CP_Z, RC_LR_Z]

            RC_LineLatLeftX = [RC_CP_X, RC_LL_X]
            RC_LineLatLeftY = [RC_CP_Y, RC_LL_Y]
            RC_LineLatLeftZ = [RC_CP_Z, RC_LL_Z]

            RC_LineMomShiftX = [RC_CP_X, RC_LR_X]
            RC_LineMomShiftY = [RC_CP_Y, RC_CP_Y]

            ################################################################################################
            #    CONDYLE PLOTS
            ################################################################################################

            # RIGHT COND SAGITTAL PLANE
            ax2 = fig.add_subplot(spec[2:4, 0:2])
            ax2.axis('equal')
            plt.gca().invert_xaxis()
            ax2.xaxis.set_major_locator(MultipleLocator(5))
            ax2.yaxis.set_major_locator(MultipleLocator(5))
            ax2.xaxis.set_minor_locator(MultipleLocator(1))
            ax2.yaxis.set_minor_locator(MultipleLocator(1))
            ax2.grid(which="minor", color="#b8b8b8", linestyle="--")
            ax2.grid(which="major", color="#959595", linestyle="--")

            ax2.patch.set_facecolor("lightskyblue")
            ax2.set_title("Sagittal Plane (right condile)")
            ax2.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
            ax2.grid(True)
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=2)
            ax2.plot(
                RC_LineOpenY,
                RC_LineOpenZ,
                color="red",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax2.plot(
                RC_LineProtrusionY,
                RC_LineProtrusionZ,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax2.plot(
                RC_LineLatRightY,
                RC_LineLatRightZ,
                color="orange",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax2.plot(
                RC_LineLatLeftY,
                RC_LineLatLeftZ,
                color="magenta",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )


            #ANGLE OF CONDYLAR GUIDANCE
            center = (RC_CP_Y, RC_CP_Z)
            p1 = np.array([(RC_CP_Y, RC_CP_Z), (RC_PR_Y, RC_CP_Z)])
            p2 = np.array([(RC_CP_Y, RC_CP_Z), (RC_PR_Y, RC_PR_Z)])

            a=np.array([RC_PR_Y, RC_CP_Z])
            b=np.array([RC_CP_Y, RC_CP_Z])
            c=np.array([RC_PR_Y, RC_PR_Z])
            ANGLE_RC_Rrot = round(np.degrees(getAngle(a, b, c)), 2)
            if RC_PR_Z>RC_CP_Z:
                am5 = AngleAnnotation(center, p1[1], p2[1], ax=ax2, size=200, text=ANGLE_RC_Rrot, linewidth=3, zorder=1)
            else:
                am5 = AngleAnnotation(center, p2[1], p1[1], ax=ax2, size=200, text=ANGLE_RC_Rrot, linewidth=3, zorder=1)


            ax = fig.add_subplot(spec[3, 0])
            image = plt.imread("images/rightright.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")

            ################################################################################################
            # LEFT COND SAGITTAL PLANE
            ax3 = fig.add_subplot(spec[2:4, 2:4])
            ax3.axis('equal')
            ax3.xaxis.set_major_locator(MultipleLocator(5))
            ax3.yaxis.set_major_locator(MultipleLocator(5))
            ax3.xaxis.set_minor_locator(MultipleLocator(1))
            ax3.yaxis.set_minor_locator(MultipleLocator(1))
            ax3.grid(which="minor", color="#b8b8b8", linestyle="--")
            ax3.grid(which="major", color="#959595", linestyle="--")

            ax3.patch.set_facecolor("antiquewhite")
            ax3.set_title("Sagittal Plane (left condile)")
            ax3.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
            ax3.grid(True)
            ax3.axhline(y=0, color="black", linestyle="--", linewidth=2)
            ax3.plot(
                LC_LineOpenY,
                LC_LineOpenZ,
                color="red",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax3.plot(
                LC_LineProtrusionY,
                LC_LineProtrusionZ,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax3.plot(
                LC_LineLatRightY,
                LC_LineLatRightZ,
                color="orange",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax3.plot(
                LC_LineLatLeftY,
                LC_LineLatLeftZ,
                color="magenta",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )

            # ANGLE OF CONDYLAR GUIDANCE
            center=(LC_CP_Y, LC_CP_Z)
            p1 = np.array([(LC_CP_Y, LC_CP_Z), (LC_PR_Y, LC_CP_Z)])
            p2 = np.array([(LC_CP_Y, LC_CP_Z), (LC_PR_Y, LC_PR_Z)])
            a=np.array([LC_PR_Y, LC_CP_Z])
            b=np.array([LC_CP_Y, LC_CP_Z])
            c=np.array([LC_PR_Y, LC_PR_Z])
            ANGLE_LC_Lrot = round(np.degrees(getAngle(a, b, c)), 2)
            if LC_PR_Z<LC_CP_Z:
                am6 = AngleAnnotation(center, p1[1], p2[1], ax=ax3, size=200, text=ANGLE_LC_Lrot, linewidth=3, zorder=1)
            else:
                am6 = AngleAnnotation(center, p2[1], p1[1], ax=ax3, size=200, text=ANGLE_LC_Lrot, linewidth=3, zorder=1)


            ax = fig.add_subplot(spec[3, 3])
            image = plt.imread("images/leftleft.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")


            # INFO PLOTS########################################################################
            axI3 = fig.add_subplot(infospec[7, 0])
            axI3.grid(False)
            plt.axis("off")

            axI3.text(0.0, -0.6, "RIGHT TMJ" + "\n" + "Angle of condylar guidance = " + str(ANGLE_RC_Rrot) + "°")

            axI3 = fig.add_subplot(infospec[7, 2])
            axI3.grid(False)
            plt.axis("off")
            axI3.text(0.0, -0.6, "LEFT TMJ" + "\n" + "Angle of condylar guidance = " + str(ANGLE_LC_Lrot) + "°")


            ################################################################################################
            # RIGHT COND TRANSVERCE PLANE
            ax4 = fig.add_subplot(spec[4:6, 0:2])
            ax4.axis('equal')
            ax4.xaxis.set_major_locator(MultipleLocator(5))
            ax4.yaxis.set_major_locator(MultipleLocator(5))
            ax4.xaxis.set_minor_locator(MultipleLocator(1))
            ax4.yaxis.set_minor_locator(MultipleLocator(1))
            ax4.grid(which="minor", color="#b8b8b8", linestyle="--")
            ax4.grid(which="major", color="#959595", linestyle="--")

            ax4.patch.set_facecolor("lightskyblue")
            ax4.set_title("Axial Plane (right condile)")
            ax4.set(xlabel="X axis, mm", ylabel="Y axis, mm")
            ax4.grid(True)
            ax4.plot(
                RC_LineOpenX,
                RC_LineOpenY,
                color="red",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax4.plot(
                RC_LineProtrusionX,
                RC_LineProtrusionY,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax4.plot(
                RC_LineLatRightX,
                RC_LineLatRightY,
                color="orange",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax4.plot(
                RC_LineLatLeftX,
                RC_LineLatLeftY,
                color="magenta",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax4.axhline(y=0, color="black", linestyle="--", linewidth=2)
            ax4.axvline(x=0, color="black", linestyle="--", linewidth=2)
            ax4.plot(
                RC_LineProtrusionX,
                RC_LineProtrusionY,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax4.plot(
                RC_LineMomShiftX,
                RC_LineMomShiftY,
                color="navy",
                linewidth=3,
                marker="D",
                mfc="white",
                mec="black",
                ms=6,
                markeredgewidth=2,
            )

            ax = fig.add_subplot(spec[5, 0])
            image = plt.imread("images/righttop.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")

            # BENNETT ANGLE
            center = (RC_CP_X, RC_CP_Y)
            p1 = np.array([(RC_CP_X, RC_CP_Y), (RC_LL_X, RC_LL_Y)])
            p2 = np.array([(RC_CP_X, RC_CP_Y), (RC_CP_X, RC_PR_Y)])

            a=np.array([RC_LL_X, RC_LL_Y])
            b=np.array([RC_CP_X, RC_CP_Y])
            c=np.array([RC_CP_X, RC_PR_Y])
            ANGLE_RBen = round(np.degrees(getAngle(a, b, c)), 2)
            if RC_LL_X<RC_CP_X:
                am7 = AngleAnnotation(center, p1[1], p2[1], ax=ax4, size=200, text=ANGLE_RBen, linewidth=3, zorder=1)
            else:
                am7 = AngleAnnotation(center, p2[1], p1[1], ax=ax4, size=200, text=ANGLE_RBen, linewidth=3, zorder=1)



            ################################################################################################
            # LEFT COND TRANSVERCE PLANE
            ax5 = fig.add_subplot(spec[4:6, 2:4])
            ax5.axis('equal')
            ax5.xaxis.set_major_locator(MultipleLocator(5))
            ax5.yaxis.set_major_locator(MultipleLocator(5))
            ax5.xaxis.set_minor_locator(MultipleLocator(1))
            ax5.yaxis.set_minor_locator(MultipleLocator(1))
            ax5.grid(which="minor", color="#b8b8b8", linestyle="--")
            ax5.grid(which="major", color="#959595", linestyle="--")

            ax5.patch.set_facecolor("antiquewhite")
            ax5.set_title("Axial Plane (left condile)")
            ax5.set(xlabel="X axis, mm", ylabel="Y axis, mm")
            ax5.grid(True)
            ax5.plot(
                LC_LineOpenX,
                LC_LineOpenY,
                color="red",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax5.plot(
                LC_LineProtrusionX,
                LC_LineProtrusionY,
                color="blue",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax5.plot(
                LC_LineLatRightX,
                LC_LineLatRightY,
                color="orange",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax5.plot(
                LC_LineLatLeftX,
                LC_LineLatLeftY,
                color="magenta",
                linewidth=3,
                marker="o",
                mfc="black",
                mec="black",
                ms=6,
            )
            ax5.axhline(y=0, color="black", linestyle="--", linewidth=2)
            ax5.axvline(x=0, color="black", linestyle="--", linewidth=2)

            ax5.plot(
                LC_LineMomShiftX,
                LC_LineMomShiftY,
                color="navy",
                linewidth=3,
                marker="D",
                mfc="white",
                mec="black",
                ms=6,
                markeredgewidth=2,
            )

            # BENNETT ANGLE
            center = (LC_CP_X, LC_CP_Y)
            p1 = np.array([(LC_CP_X, LC_CP_Y), (LC_LR_X, LC_LR_Y)])
            p2 = np.array([(LC_CP_X, LC_CP_Y), (LC_CP_X, LC_PR_Y)])

            a=np.array([LC_LR_X, LC_LR_Y])
            b=np.array([LC_CP_X, LC_CP_Y])
            c=np.array([LC_CP_X, LC_PR_Y])
            ANGLE_LBen = round(np.degrees(getAngle(a, b, c)), 2)
            if LC_LR_X<LC_CP_X:
                am8 = AngleAnnotation(center, p1[1], p2[1], ax=ax5, size=200, text=ANGLE_LBen, linewidth=3, zorder=1)
            else:
                am8 = AngleAnnotation(center, p2[1], p1[1], ax=ax5, size=200, text=ANGLE_LBen, linewidth=3, zorder=1)


            ax = fig.add_subplot(spec[5, 3])
            image = plt.imread("images/lefttop.png")
            ax.imshow(image, alpha=headimgalpha)
            ax.axis("off")

            # INFO PLOTS########################################################################
            RightShift = abs(round(RC_LR_X - RC_CP_X, 3))
            LeftShift = abs(round(LC_LL_X - LC_CP_X, 3))

            axI5 = fig.add_subplot(infospec[11, 0])
            axI5.grid(False)
            plt.axis("off")
            axI5.text(
                0.0,
                -1,
                "RIGHT TMJ"
                + "\n"
                + "Bennett angle = "
                + str(ANGLE_RBen)
                + "°"        
                + "\n"
                + "Bennett shift = "
                + str(RightShift)
                + " mm",
            )

            axI6 = fig.add_subplot(infospec[11, 2])
            axI6.grid(False)
            plt.axis("off")
            axI6.text(
                0.0,
                -1,
                "LEFT TMJ"
                + "\n"
                + "Bennett angle = "
                + str(ANGLE_LBen)
                + "°"
                + "\n"
                + "Bennett shift = "
                + str(LeftShift)
                + " mm",
            )

            st.pyplot(fig)

if __name__ == "__main__":
    main()