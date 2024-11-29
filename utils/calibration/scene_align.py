import argparse
import os
import json

import numpy as np
import logging
from datetime import datetime

try:
    from aruco_detect import ArucoDetector
except Exception as e:
    print(e)
try:
    from apriltag_detect import ApriTagDetector
except Exception as e:
    print(e)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2


def open_camera(cam_list, max_id=10):
    if cam_list == []:
        current = 0
        while current < max_id:
            cap = cv2.VideoCapture(current)
            # cap = mvcamera.VideoCapture(current)
            if cap.isOpened():
                cam_list.append(current)
            if cap is not None:
                # cap.release()
                pass
            current += 1

        if cam_list == []:
            print("No available camera found")
            exit()
        print(f"Cameras found: {cam_list}")

    cap = cv2.VideoCapture(cam_list[0])
    # cap = mvcamera.VideoCapture(cam_list[0])
    if cap.isOpened():
        print(f"Opened camera {cam_list[0]}")
    else:
        print(f"Camera: {cam_list[0]} failed to open")
        exit()
    cam_list.append(cam_list.pop(0))
    return cap


def load_config(config_path):
    if config_path is None:
        print("No config file provided; use None for all values")
        config = {
            "reference_pose": None,
            "reference_image": None,
            "camera_ref": None,
            "cameras_intri_dict": None,
            "tag": {
                "size": None,
                "families": None,
            },
        }
        return config

    with open(config_path, "r") as file:
        config = json.load(file)
        for value in config.values():
            value = np.array(value)
        reference_pose = config["reference_pose"]
        reference_image = cv2.imread(config["reference_image"][camera_ref])
        # refs = (reference_image, reference_pose)
        # assert refs != (
        #     None,
        #     None,
        # ), "You must provide either a reference image or a reference pose"
        config["reference_pose"] = reference_pose
        config["reference_image"] = reference_image
    return config


def ncc(imageA, imageB):
    """Normalized Cross-Correlation (NCC) calculation"""
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"

    meanA = np.mean(imageA)
    meanB = np.mean(imageB)

    ncc_value = np.sum((imageA - meanA) * (imageB - meanB))
    ncc_value /= np.sqrt(np.sum((imageA - meanA) ** 2) * np.sum((imageB - meanB) ** 2))

    return ncc_value


def calculate_pose_difference(current_pose, reference_pose):
    translation_diff = current_pose[:3, 3] - reference_pose[:3, 3]
    rotation_diff = current_pose[:3, :3] @ np.linalg.inv(reference_pose[:3, :3])

    # Actually cv2.Rodrigues returns a rotation vector, not Euler angles
    rotation_diff_euler = cv2.Rodrigues(rotation_diff)[0].flatten()

    return translation_diff, rotation_diff_euler


def normalize_array(arr, val):
    for i in range(len(arr)):
        arr[i] = np.clip(arr[i] / val, -1, 1)


def draw_adjustment_guidance(frame, translation_diff, rotation_diff_euler, metric=None):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    scale = int(h / 3)
    edge_dist = 50
    marker_radius = 5
    line_thickness = 2
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # Translation adjustment indicator (green arrow)
    arrow_length = int(np.linalg.norm(translation_diff[:2]) * scale)
    angle = np.arctan2(translation_diff[1], translation_diff[0])
    end_point = (
        int(center[0] + arrow_length * np.cos(angle)),
        int(center[1] + arrow_length * np.sin(angle)),
    )
    cv2.arrowedLine(frame, center, end_point, GREEN, line_thickness)

    # Z-axis movement indicator
    z_arrow_length = int(abs(translation_diff[2]) * scale)
    z_direction = -1 if translation_diff[2] > 0 else 1
    z_start_point = (edge_dist, center[1])
    z_end_point = (edge_dist, center[1] + z_direction * z_arrow_length)
    cv2.arrowedLine(frame, z_start_point, z_end_point, GREEN, line_thickness)
    cv2.putText(frame, "Z", (10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

    # Rotation indicator lines (pitch, yaw, roll) in red
    yaw_line_start = (center[0] - scale, center[1])
    yaw_line_end = (center[0] + scale, center[1])
    cv2.line(frame, yaw_line_start, yaw_line_end, RED, line_thickness, cv2.LINE_AA)
    text_pos = (center[0] + scale + 20, center[1])
    cv2.putText(
        frame, "Yaw", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA
    )
    yaw_marker = np.array([int(center[0] + rotation_diff_euler[1] * scale), center[1]])
    marker_pos = (yaw_marker - [0, marker_radius], yaw_marker + [0, marker_radius])
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    pitch_line_start = (center[0], center[1] - scale)
    pitch_line_end = (center[0], center[1] + scale)
    cv2.line(frame, pitch_line_start, pitch_line_end, RED, line_thickness)
    text_pos = (center[0] + 20, center[1] - scale - 10)
    cv2.putText(
        frame, "Pitch", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA
    )
    pitch_marker = np.array(
        [center[0], int(center[1] - rotation_diff_euler[0] * scale)]
    )
    marker_pos = (pitch_marker - [marker_radius, 0], pitch_marker + [marker_radius, 0])
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    tilt_scale = int(scale / np.sqrt(2))
    roll_line_start = (center[0] - tilt_scale, center[1] - tilt_scale)
    roll_line_end = (center[0] + tilt_scale, center[1] + tilt_scale)
    cv2.line(frame, roll_line_start, roll_line_end, RED, line_thickness)
    text_pos = (center[0] + tilt_scale + 20, center[1] + tilt_scale + 20)
    cv2.putText(
        frame, "Roll", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA
    )
    roll_marker = np.array(
        [
            int(center[0] + rotation_diff_euler[2] * tilt_scale),
            int(center[1] + rotation_diff_euler[2] * tilt_scale),
        ]
    )
    tilt_marker_radius = (
        np.array([marker_radius, -marker_radius]) / np.sqrt(2)
    ).astype(int)
    marker_pos = (roll_marker - tilt_marker_radius, roll_marker + tilt_marker_radius)
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    if metric is not None:
        cv2.putText(
            frame,
            f"NCC: {metric:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            BLUE,
            2,
            cv2.LINE_AA,
        )


def to_pose(tvec, Rmat):
    if tvec is None or Rmat is None:
        return None

    pose = np.eye(4)
    pose[:3, 3] = tvec.flatten()
    pose[:3, :3] = Rmat
    return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_path", type=str, required=True)
    parser.add_argument("-c", "--camera_ids", type=int, nargs="+", default=[0])
    parser.add_argument("-f", "--fps", type=int, default=30)
    # TODO 值越低标记越灵敏
    parser.add_argument("-ss", "--sensitivity", type=float, default=0.2)
    parser.add_argument("-sd", "--save_dir", type=str)
    parser.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="Overwrite existing config files",
    )
    args = parser.parse_args()

    camera_ids = args.camera_ids
    sensitivity = args.sensitivity
    save_dir = (
        os.path.dirname(args.config_path) if args.save_dir is None else args.save_dir
    )

    if args.config_path is None:
        print("No config file provided; use None for all values")
    else:
        with open(args.config_path, "r") as file:
            config = json.load(file)
            camera_ref=config["camera_ref"][str(camera_ids[0])]

    config = load_config(args.config_path)
    cameras_intri_dict=config["cameras_intri_dict"]
    camera_ref_intri_dict = config["cameras_intri_dict"][camera_ref]
    camera_ref_intri_dict["camera_matrix"] = np.array(camera_ref_intri_dict["camera_matrix"], dtype=np.float32)
    camera_ref_intri_dict["dist_coeffs"] = np.array(camera_ref_intri_dict["dist_coeffs"], dtype=np.float32)
    tag_size = config["tag"]["size"]
    tag_families = config["tag"]["families"]
    reference_image = config["reference_image"]
    if config["reference_pose"] is not None:
        reference_pose = config["reference_pose"][camera_ref]
    else:
        reference_pose = None
    #print(reference_pose)
    #input("wait")

    # D455 RGB
    # camera_params = [384.904, 384.387, 321.351, 243.619]  # [fx, fy, cx, cy]
    # camera_width = 1280
    # camera_height = 800
    if isinstance(tag_families, int):
        detector = ArucoDetector(tag_families)
    elif "tag" in tag_families:
        # options = apriltag.DetectorOptions(families=tag_families)
        # detector = apriltag.Detector(options)
        detector = ApriTagDetector(tag_families)
    else:
        raise ValueError(f"Invalid tag family: {tag_families}")

    # detect pose from reference image if provided
    if reference_image is not None:
        if reference_pose != None:
            print(
                "Both reference image and reference, pose provided, using reference pose"
            )
            reference_image = None
        else:
            detector.set_camera_params(camera_ref, camera_ref_intri_dict["camera_matrix"], camera_ref_intri_dict["dist_coeffs"])
            result = detector.detect(
                reference_image,
                camera_ref,
                config["tag"]["size"],
            )
            # TODO: implement to pose
            if "tvecs" in result and len(result["tvecs"]) > 0 and "Rmats" in result and len(result["Rmats"]) > 0:
                reference_pose = to_pose(result["tvecs"][0], result["Rmats"][0])
                #print(reference_pose)
                #input("wait")
            if reference_pose is None:
                print("Failed to detect AprilTag/ArucoTag in reference image")
            else:
                print(f"Reference image provided, detected pose: {reference_pose}")

    # TODO: support for multiple cameras simultaneously processing and displaying
    cap = open_camera(camera_ids)
    cv2.namedWindow("Current Camera View", cv2.WINDOW_KEEPRATIO)
    overlay = False



    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(
                f"Failed to read frame from camera {camera_ids[0]}: {datetime.now().timestamp()}"
            )
            continue
        
        camera = str(camera_ids[0])
        detector.set_camera_params(camera, cameras_intri_dict[camera]["camera_matrix"],cameras_intri_dict[camera]["dist_coeffs"])
        
        result = detector.detect(frame, camera, tag_size)

        if "tvecs" in result and len(result["tvecs"]) > 0 and "Rmats" in result and len(result["Rmats"]) > 0:
          
            current_pose = to_pose(result["tvecs"][0], result["Rmats"][0])
        else:
            current_pose = None
            
        


        ncc_value = None
        if overlay:
            if reference_image is not None:
                frame = np.mean([reference_image, frame], axis=0).astype(np.uint8)
                ncc_value = ncc(reference_image, frame)

        if current_pose is not None:
            if reference_pose is not None:

                translation_diff, rotation_diff_euler = calculate_pose_difference(
                    np.array(current_pose), np.array(reference_pose)
                )
                normalize_array(translation_diff, sensitivity)
                normalize_array(rotation_diff_euler, sensitivity)
                draw_adjustment_guidance(
                    frame, translation_diff, rotation_diff_euler, ncc_value
                )

        cv2.imshow("Current Camera View", frame)

        print(
            "q: quit, o: toggle overlay, s: save new reference image, n: switch camera"
        )
        key = cv2.waitKey(1)
        print(ord("q"), ord("o"), ord("s"), ord("n"))
        if key == ord("q") or key == 27:
            break
        elif key == ord("o"):
            overlay = not overlay
        elif key == ord("s"):
            # save current frame as reference image and its pose
            assert camera_ref_intri_dict is not None, "camera_ref_intri_dict must be provided"
            assert tag_size is not None, "Tag size must be provided"            
            reference_pose = current_pose
            if reference_pose is None:  
             input("no ArUcoTag found")
             continue      

            save_config = {
                "reference_pose":reference_pose.tolist(),
            }

            if not args.overwrite:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if reference_pose is not None:
                    file_name = f"config_{time_str}.json"
                    with open(f"{save_dir}/{file_name}", "w") as file:
                        json.dump(save_config, file)
                    print(f"Reference pose saved to {save_dir}/{file_name}")
                    no_pose = ""
                else:
                    file_name = None
                    print("Failed to detect AprilTag/ArUcoTag in current frame")
                    no_pose = "no_pose_"
                reference_image = frame
              
                img_name = f"reference_image_{no_pose}{time_str}.jpg"
                cv2.imwrite(f"{save_dir}/{img_name}", reference_image)
                print(f"Reference image saved to {save_dir}/{img_name}")
                
        elif key == ord("n"):
            # TODO: remove this
            camera_ref=config["camera_ref"][str(camera_ids[0])]
          
            cap = open_camera(camera_ids)
            config = load_config(args.config_path)
            cameras_intri_dict=config["cameras_intri_dict"]
            camera_ref_intri_dict = config["cameras_intri_dict"][camera_ref]
            camera_ref_intri_dict["camera_matrix"] = np.array(camera_ref_intri_dict["camera_matrix"], dtype=np.float32)
            camera_ref_intri_dict["dist_coeffs"] = np.array(camera_ref_intri_dict["dist_coeffs"], dtype=np.float32)
            tag_size = config["tag"]["size"]
            tag_families = config["tag"]["families"]
            reference_image = config["reference_image"]

            if config["reference_pose"] is not None:
                reference_pose = config["reference_pose"][camera_ref]
            else :
                reference_pose= None
    #print(reference_pose)
    #input("wait")

    # D455 RGB
    # camera_params = [384.904, 384.387, 321.351, 243.619]  # [fx, fy, cx, cy]
    # camera_width = 1280
    # camera_height = 800


    # detect pose from reference image if provided
            if reference_image is not None:
                if reference_pose is not None:
                    print(
                        "Both reference image and reference, pose provided, using reference pose"
                    )
                    reference_image = None
                else:
                    detector.set_camera_params(camera_ref, camera_ref_intri_dict["camera_matrix"], camera_ref_intri_dict["dist_coeffs"])
                    result = detector.detect(
                        reference_image,
                        camera_ref,
                        config["tag"]["size"],
                        )
                    if "tvecs" in result and len(result["tvecs"]) > 0 and "Rmats" in result and len(result["Rmats"]) > 0:
                        reference_pose = to_pose(result["tvecs"][0], result["Rmats"][0])
                    if reference_pose is None:
                        print("Failed to detect AprilTag/ArucoTag in reference image")
                    else:
                        print(f"Reference image provided, detected pose: {reference_pose}")


            
            #input("abc")       
            #reference_image = load_config(args.config_path)["reference_image"]
            #reference_pose = load_config(args.config_path)["reference_pose"]

    cap.release()
    cv2.destroyAllWindows()
