import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random

from navid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from navid.conversation import conv_templates, SeparatorStyle
from navid.model.builder import load_pretrained_model
from navid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from std_msgs.msg import Int32
import tf
import math
from ultralytics import YOLO
from openai import OpenAI
import re
import difflib
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
from tf.transformations import euler_from_quaternion

import dedistortion
from PIL import Image as PILImage

def evaluate_agent(config, split_id, dataset, model_path, result_path) -> None:

    env = Env(config.TASK_CONFIG, dataset)

    agent = NaVid_Agent(model_path, result_path)

    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0
    
      
    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id)):
        obs = env.reset()
        iter_step = 0
        agent.reset()

         
        continuse_rotation_count = 0
        last_dtg = 999
        while not env.episode_over:
            
            info = env.get_metrics()
            
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
            
            
            action = agent.act(obs, info, env.current_episode.episode_id)
            
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}

            
            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1



        with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(result_dict, f, indent=4)


class VLM:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.state_sub = rospy.Subscriber("/state_estimation", Odometry, self.state_callback)
        self.question_sub = rospy.Subscriber("/challenge_question", String, self.question_callback, queue_size=5)
        self.explore_sub = rospy.Subscriber("/way_point_with_heading_explore", Pose2D, self.explore_callback, queue_size=5)

        # 比赛官方指定的三个topic
        self.num_pub = rospy.Publisher("/numerical_response", Int32, queue_size=5)
        self.marker_pub = rospy.Publisher("/selected_object_marker", Marker, queue_size=5)
        self.waypoint_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=5)
        # 新增：YOLO 结果图发布者（RViz 订阅这个看拉框）
        self.yolo_img_pub = rospy.Publisher('/yolo_annotated_image', Image, queue_size=1)
        self.cropped_image_pub = rospy.Publisher('/cropped_image', Image, queue_size=1)

        self.subtasks = []
        self.subtask_idx = 0
        self.category = None
        self.keyword = None

        self.bridge = CvBridge()
        self.agent = None
        self.run_once = False
        self.obs = {
            "rgb": None,
            "instruction": {}
        }
        self.action = None
        self.yaw = 0.
        self.msg = Pose2D()
        self.stop_count = 0
        self.done = False

        # 新增：YOLO 懒加载句柄 + 控制只发布一次
        self.yolo = None
        self.yolo_published = False

        # 应对第一类问题
        self.num = Int32()
        self.obj_gallery = []

        # 应对第二类问题
        self.obj_box = None
        self.marker = Marker()

        # 探索
        self.explore_x = 0.
        self.explore_y = 0.
        self.explore_theta = 0.

        random.seed(42)

    def explore_callback(self, msg):
        self.explore_x = msg.x
        self.explore_y = msg.y
        self.explore_theta = msg.theta

    def image_callback(self, msg):
        # 原图（BGR）
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        output_image = dedistortion.panorama_to_plane(img_bgr, 90, (480, 480), 270, 90)

        # 统一转成 RGB 
        if isinstance(output_image, PILImage.Image):
            out_rgb = np.array(output_image)             
        elif isinstance(output_image, np.ndarray):
            out_rgb = output_image
            # 如果返回的是 BGR，这里转一下：
            # out_rgb = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError("output_image must be PIL.Image.Image or np.ndarray")

        # 发布到 /cropped_image （rgb8）
        ros_img = self.bridge.cv2_to_imgmsg(out_rgb, encoding="rgb8")
        ros_img.header = msg.header
        self.cropped_image_pub.publish(ros_img)

        self.obs["rgb"] = out_rgb

    def state_callback(self, msg):
        self.pos = msg.pose.pose.position
        self.ori = msg.pose.pose.orientation

        # 四元数转欧拉角
        orientation_q = self.ori
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.yaw = yaw

    def question_callback(self, msg):
        self.obs["instruction"]["text"] = msg.data

    def test(self) -> None:
        return None

    def evaluate_agent_CMU(self, model_path, result_path) -> None:   
        if not isinstance(self.obs.get("instruction"), dict):
            return None
        
        # 只运行一次的部分 =======================================================================================================================================
        if self.run_once == False:
            self.agent = NaVid_Agent(model_path, result_path, require_map=False)
            self.agent.reset()
            
            # 拆解指令 ==========================================================================================================================================
            os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

            instr = self.obs.get("instruction") or {}
            prompt_text = instr.get("text")
            if not prompt_text:
                rospy.logwarn("GPT: instruction['text'] is empty, skipping this round.")
            else:
                client = OpenAI()

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You parse navigation instructions.\n"
                            "Process:\n"
                            "Step 1) Classify the input into one of: Numerical (1) / Object Reference (2) / Instruction-Following (3).\n"
                            "Step 2) Output exactly SEVEN brace slots in ONE line: {}{}{}{}{}{}{} .\n"
                            "Output spec (strict):\n"
                            "Slot1 = {category_id} where 1=Numerical, 2=Object Reference, 3=Instruction-Following.\n"
                            "Slot2 = {keyword} the core object noun in singular form (e.g., 'chair', 'potted plant'); use 'none' ONLY for Instruction-Following.\n"
                            "Slots3-7 = five action/location instructions. Each slot contains either 'none' or ONE English imperative sentence "
                            "like 'Go near ...', 'Go to ...', 'Take the path ...', 'Stop at ...'.\n"
                            "Fill left to right; unused slots must be 'none'.\n"
                            "Print NOTHING except these seven brace slots in one line. No category names, no explanations, no newlines.\n"
                            # Category templates
                            "Category templates:\n"
                            "• Numerical (1): Identify target & reference. Slot3 uses the base action; Slots4-7 repeat the same action but with the word 'another' "
                            "to emphasize a different instance, e.g., 'Go near another chair ...'.\n"
                            "• Object Reference (2): Slot3 is the action to approach that object; Slots4-7 are 'none'.\n"
                            "• Instruction-Following (3): Split sequential actions into steps; fill Slot3 with step1, Slot4 with step2, etc.; remaining slots 'none'.\n"
                        )
                    },

                    # Numerical
                    {"role": "user", "content": "How many blue chairs are between the table and the wall?"},
                    {"role": "assistant", "content":
                        "{1}"
                        "{chair}"
                        "{Go near the chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                    },

                    # Object Reference
                    {"role": "user", "content": "Find the potted plant on the kitchen island that is closest to the fridge."},
                    {"role": "assistant", "content":
                        "{2}"
                        "{potted plant}"
                        "{Go near the potted plant on the kitchen island that is closest to the fridge.}"
                        "{none}{none}{none}{none}"
                    },

                    # Instruction-Following
                    {"role": "user", "content": "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table."},
                    {"role": "assistant", "content":
                        "{3}"
                        "{none}"
                        "{Go to the potted plant furthest from the hookah.}"
                        "{Take the path between the two columns, and stop at the tray on the table.}"
                        "{none}{none}{none}"
                    },

                    # Input
                    {"role": "user", "content": f"{prompt_text}"},
                ]

                completion = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=220, messages=messages)
                out = (completion.choices[0].message.content or "").strip()

            self.subtasks = re.findall(r"\{([^{}]*)\}", out)  # expect 7
            self.subtasks = [(s or "").strip() for s in self.subtasks]
            if len(self.subtasks) < 7:
                self.subtasks += ["none"] * (7 - len(self.subtasks))
            elif len(self.subtasks) > 7:
                self.subtasks = self.subtasks[:7]

            cat_raw = self.subtasks[0]
            self.keyword = self.subtasks[1] or "none"
            self.subtasks = [(a or "none").strip() or "none" for a in self.subtasks[2:7]]

            try:
                self.category = int(cat_raw)
                if self.category not in (1, 2, 3):
                    raise ValueError
            except Exception:
                self.category = 0

            rospy.loginfo(f"{prompt_text}")
            rospy.loginfo(f"category={self.category}, keyword={self.keyword}, actions={self.subtasks}")

            self.run_once = True

        # 所有子任务都完成了，持续发送类型1和类型2对应的话题 ==========================================================================================================
        if self.subtask_idx >= len(self.subtasks):
            self.marker_pub.publish(self.marker)
            self.num.data  = len(self.obj_gallery)
            self.num_pub.publish(self.num)
            rospy.loginfo(f"Published numerical_response={self.num}")
            return None
        
        # 跳过所有"none"任务 ====================================================================================================================================
        if "none" in self.subtasks[self.subtask_idx].lower():
            self.marker_pub.publish(self.marker)
            self.num.data  = len(self.obj_gallery)
            self.num_pub.publish(self.num)
            rospy.loginfo(f"Published numerical_response={self.num}")
            self.subtask_idx += 1
            return None
        
        # 依次将拆解后的指令送入Navid =============================================================================================================================
        self.obs["instruction"]["text"] = self.subtasks[self.subtask_idx]
        rospy.loginfo(str(self.obs["instruction"]["text"]))
        result = self.agent.act(self.obs, None, "CMU")
        self.action = result["action"]

        # 如果连续出现5次0，永远不再发布msg，认为任务已经完成 =========================================================================================================
        if self.action == 0:
            self.stop_count += 1
        else:
            self.stop_count = 0
        if self.stop_count >= 5:
            self.done = True

        if self.done == False:
            if self.action == 0:
                # 停住
                None
            elif self.action == 1:
                # 直行25cm
                self.msg.x = self.pos.x + 0.25 * math.cos(self.yaw)
                self.msg.y = self.pos.y + 0.25 * math.sin(self.yaw)
            elif self.action == 2:
                # 左转30度
                self.msg.theta = self.yaw + math.radians(30)
            elif self.action == 3:
                # 右转30度
                self.msg.theta = self.yaw - math.radians(30)
            else:
                # 停住
                None
            self.waypoint_pub.publish(self.msg)
        
        if self.done == True:
            self.subtask_idx += 1  
            self.done = False
            self.stop_count = 0

            # 用YOLO拉框，并通过话题发送（每个 subtask 完成后执行一次）
            if self.obs["rgb"] is None:
                rospy.logwarn("YOLO: obs['rgb'] is None, skip.")
                return

            # 加载YOLO
            if self.yolo is None:
                try:
                    self.yolo = YOLO('yolov8n.pt')
                    rospy.loginfo("YOLO model loaded.")
                except Exception as e:
                    rospy.logerr("Failed to load YOLO model: %s", str(e))
                    return

            try:
                # 1) YOLO 推理
                bgr = self.obs["rgb"]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = self.yolo.predict(source=rgb, verbose=False)

                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    rospy.logwarn("YOLO: no detections.")
                    return

                res = results[0]
                names = res.names  # id -> label 映射
                cls_ids = res.boxes.cls.tolist()
                labels = [names[int(c)] for c in cls_ids]
                unique_labels = sorted(set(labels))

                # 保留与keyeord最匹配的label（可能有多个）==========================================================================================================
                selected_label = None
                keyword = (self.keyword or "none").strip()
                if self.category in (1, 2) and len(unique_labels) > 1 and keyword.lower() != "none":
                    try:
                        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
                        os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
                        client = OpenAI()
                        options_line = ", ".join(unique_labels)

                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a label selector. Your goal is to choose EXACTLY ONE label "
                                    "from the candidate list that best matches the given keyword.\n"
                                    "Strict rules:\n"
                                    "1) Always return EXACTLY one label FROM THE CANDIDATE LIST, spelled EXACTLY as given.\n"
                                    "2) Prefer the candidate that matches the SAME SEMANTIC TYPE as the keyword "
                                    "(e.g., lighting/fixture vs plant vs furniture vs appliance vs container, etc.).\n"
                                    "3) If multiple candidates share the same type, choose the one semantically closest to the keyword "
                                    "(synonyms, hyponyms/hypernyms accepted: 'refrigerator' ≈ 'fridge', 'bin' ≈ 'trash can').\n"
                                    "4) If none share the same type, choose the closest by meaning; break ties with string similarity.\n"
                                    "5) Output ONLY the chosen label. Do not explain, do not add quotes or extra text."
                                )
                            },

                            {"role": "user", "content": "keyword: wall lamp\ncandidates: potted plant, traffic light\nChoose ONE from candidates."},
                            {"role": "assistant", "content": "traffic light"},

                            {"role": "user", "content": "keyword: refrigerator\ncandidates: fridge, cabinet\nChoose ONE from candidates."},
                            {"role": "assistant", "content": "fridge"},

                            {"role": "user", "content": "keyword: chair\ncandidates: sofa, computer, table\nChoose ONE from candidates."},
                            {"role": "assistant", "content": "sofa"},

                            {"role": "user", "content": "keyword: plant\ncandidates: potted plant, traffic light\nChoose ONE from candidates."},
                            {"role": "assistant", "content": "potted plant"},

                            {"role": "user", "content": "keyword: bin\ncandidates: trash can, vase\nChoose ONE from candidates."},
                            {"role": "assistant", "content": "trash can"},

                            {
                                "role": "user",
                                "content": f"keyword: {keyword}\ncandidates: {options_line}\nChoose ONE from candidates."
                            },
                        ]

                        completion = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=10, messages=messages)
                        raw = (completion.choices[0].message.content or "").strip()
                        cand = raw.strip("'").strip('"')

                        if cand in unique_labels:
                            selected_label = cand
                        else:
                            # GPT返回不在候选中：用相似度兜底 ======================================================================================================
                            best = difflib.get_close_matches(keyword.lower(), [u.lower() for u in unique_labels], n=1, cutoff=0.0)
                            selected_label = (
                                unique_labels[[u.lower() for u in unique_labels].index(best[0])]
                                if best else unique_labels[0]
                            )

                        rospy.loginfo(f"Label selection -> keyword='{keyword}', chosen='{selected_label}', candidates={unique_labels}")

                    except Exception as e:
                        rospy.logwarn(f"GPT label selection failed, fallback to heuristic: {e}")
                        best = difflib.get_close_matches(keyword.lower(), [u.lower() for u in unique_labels], n=1, cutoff=0.0)
                        selected_label = (
                            unique_labels[[u.lower() for u in unique_labels].index(best[0])]
                            if best else unique_labels[0]
                        )
                else:
                    # 非1/2类或只有1个候选或keyword无效：直接用唯一/第一个
                    selected_label = unique_labels[0]

                # 只保留选中标签中置信度最高的那个框 ================================================================================================================
                conf_list = res.boxes.conf.tolist()
                indices = [i for i, lbl in enumerate(labels) if lbl == selected_label]

                if indices:
                    # 选置信度最大的那个索引
                    best_idx = max(indices, key=lambda i: conf_list[i] if i < len(conf_list) else -1.0)

                    # 只保留该 index 的框
                    keep_mask = [i == best_idx for i in range(len(labels))]
                    res.boxes = res.boxes[keep_mask]

                    # 日志里也只显示这个框
                    labels = [labels[best_idx]]

                # 4) 可视化并发布
                annotated_bgr = res.plot()
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
                annotated_msg.header.stamp = rospy.Time.now()
                annotated_msg.header.frame_id = "map"
                self.yolo_img_pub.publish(annotated_msg)

                rospy.loginfo(f"YOLO published with label='{selected_label}', kept {len(labels)} boxes.")

                # 假设只保留了一个 box
                self.obj_box = res.boxes[0]

                # xyxy 格式 [x1, y1, x2, y2]，是像素坐标
                x1, y1, x2, y2 = self.obj_box.xyxy[0].tolist()

                # 裁剪图像
                crop = bgr[int(y1):int(y2), int(x1):int(x2)].copy()

                # 存入数组
                self.obj_gallery.append(crop)
                rospy.loginfo(f"Stored object crop #{len(self.obj_gallery)} with label='{selected_label}'")

                # TODO：对self.obj_gallery中的元素比对去重 ========================================================================

                # 中心点（像素坐标）
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # 边界框的宽高
                objW = abs(x2 - x1)
                objH = abs(y2 - y1)

                # 映射比例
                scale = 0.01
                objMidX = cx * scale
                objMidY = cy * scale
                objMidZ = 0.0   # 图像是 2D 的，默认 0

                # Marker 消息
                self.marker.header.frame_id = "map"
                self.marker.header.stamp = rospy.Time.now()
                self.marker.ns = selected_label
                self.marker.id = 0
                self.marker.type = Marker.CUBE
                self.marker.action = Marker.ADD

                # 位置
                self.marker.pose.position.x = objMidX
                self.marker.pose.position.y = objMidY
                self.marker.pose.position.z = objMidZ

                # 发布 Marker
                self.marker_pub.publish(self.marker)

                rospy.loginfo(
                    f"Published CUBE marker for {selected_label} at "
                    f"({objMidX:.2f}, {objMidY:.2f}, {objMidZ:.2f}) "
                    f"size=({self.marker.scale.x:.2f}, {self.marker.scale.y:.2f}, {self.marker.scale.z:.2f})"
                )

            except Exception as e:
                rospy.logerr("YOLO inference failed: %s", str(e))

        self.num.data  = len(self.obj_gallery)
        self.num_pub.publish(self.num)
        rospy.loginfo(f"Published numerical_response={self.num}")

    def evaluate_agent_CMU_2(self, model_path, result_path) -> None:   
        if not isinstance(self.obs.get("instruction"), dict):
            return None
        
        # 只运行一次的部分 =======================================================================================================================================
        if self.run_once == False:
            self.agent = NaVid_Agent(model_path, result_path, require_map=False)
            self.agent.reset()
            
            # 拆解指令 ==========================================================================================================================================
            os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

            instr = self.obs.get("instruction") or {}
            prompt_text = instr.get("text")
            if not prompt_text:
                rospy.logwarn("GPT: instruction['text'] is empty, skipping this round.")
            else:
                client = OpenAI()

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You parse navigation instructions.\n"
                            "Process:\n"
                            "Step 1) Classify the input into one of: Numerical (1) / Object Reference (2) / Instruction-Following (3).\n"
                            "Step 2) Output exactly SEVEN brace slots in ONE line: {}{}{}{}{}{}{} .\n"
                            "Output spec (strict):\n"
                            "Slot1 = {category_id} where 1=Numerical, 2=Object Reference, 3=Instruction-Following.\n"
                            "Slot2 = {keyword} the core object noun in singular form (e.g., 'chair', 'potted plant'); use 'none' ONLY for Instruction-Following.\n"
                            "Slots3-7 = five action/location instructions. Each slot contains either 'none' or ONE English imperative sentence "
                            "like 'Go near ...', 'Go to ...', 'Take the path ...', 'Stop at ...'.\n"
                            "Fill left to right; unused slots must be 'none'.\n"
                            "Print NOTHING except these seven brace slots in one line. No category names, no explanations, no newlines.\n"
                            # Category templates
                            "Category templates:\n"
                            "• Numerical (1): Identify target & reference. Slot3 uses the base action; Slots4-7 repeat the same action but with the word 'another' "
                            "to emphasize a different instance, e.g., 'Go near another chair ...'.\n"
                            "• Object Reference (2): Slot3 is the action to approach that object; Slots4-7 are 'none'.\n"
                            "• Instruction-Following (3): Split sequential actions into steps; fill Slot3 with step1, Slot4 with step2, etc.; remaining slots 'none'.\n"
                        )
                    },

                    # Numerical
                    {"role": "user", "content": "How many blue chairs are between the table and the wall?"},
                    {"role": "assistant", "content":
                        "{1}"
                        "{chair}"
                        "{Go near the chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                        "{Go near another chair that is between the table and the wall.}"
                    },

                    # Object Reference
                    {"role": "user", "content": "Find the potted plant on the kitchen island that is closest to the fridge."},
                    {"role": "assistant", "content":
                        "{2}"
                        "{potted plant}"
                        "{Go near the potted plant on the kitchen island that is closest to the fridge.}"
                        "{none}{none}{none}{none}"
                    },

                    # Instruction-Following
                    {"role": "user", "content": "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table."},
                    {"role": "assistant", "content":
                        "{3}"
                        "{none}"
                        "{Go to the potted plant furthest from the hookah.}"
                        "{Take the path between the two columns, and stop at the tray on the table.}"
                        "{none}{none}{none}"
                    },

                    # Input
                    {"role": "user", "content": f"{prompt_text}"},
                ]

                completion = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=220, messages=messages)
                out = (completion.choices[0].message.content or "").strip()

            self.subtasks = re.findall(r"\{([^{}]*)\}", out)  # expect 7
            self.subtasks = [(s or "").strip() for s in self.subtasks]
            if len(self.subtasks) < 7:
                self.subtasks += ["none"] * (7 - len(self.subtasks))
            elif len(self.subtasks) > 7:
                self.subtasks = self.subtasks[:7]

            cat_raw = self.subtasks[0]
            self.keyword = self.subtasks[1] or "none"
            self.subtasks = [(a or "none").strip() or "none" for a in self.subtasks[2:7]]

            try:
                self.category = int(cat_raw)
                if self.category not in (1, 2, 3):
                    raise ValueError
            except Exception:
                self.category = 0

            rospy.loginfo(f"{prompt_text}")
            rospy.loginfo(f"category={self.category}, keyword={self.keyword}, actions={self.subtasks}")

            self.run_once = True

        # 第一类问题 ===========================================================================================================================================
        if self.category >= 0.9 and self.category <= 1.1:
            # # 乱走 
            # self.msg.x = self.explore_x
            # self.msg.y = self.explore_y
            # self.msg.theta = self.explore_theta
            # self.waypoint_pub.publish(self.msg)  

            # # TODO - 建图
                
            # # TODO - 基于关键词遍历列表，去重，发送话题
            # self.keyword 
            # None

            num = random.uniform(1, 5)
            self.num_pub.publish(num)

        
        # 第二类和第三类问题 ===========================================================================================================================================
        elif (self.category >= 1.9 and self.category <= 2.1) or (self.category >= 2.9 and self.category <= 3.1):
            # 所有子任务都完成了，持续发送第二类问题对应的话题 ==========================================================================================================
            if self.subtask_idx >= len(self.subtasks):
                if (self.category >= 1.9 and self.category <= 2.1): 
                    self.marker_pub.publish(self.marker)
                return None
            
            # 跳过所有"none"任务 ====================================================================================================================================
            if "none" in self.subtasks[self.subtask_idx].lower():
                self.marker_pub.publish(self.marker)
                self.subtask_idx += 1
                return None
            
            # 依次将拆解后的指令送入Navid =============================================================================================================================
            self.obs["instruction"]["text"] = self.subtasks[self.subtask_idx]
            rospy.loginfo(str(self.obs["instruction"]["text"]))
            result = self.agent.act(self.obs, None, "CMU")
            self.action = result["action"]

            # 如果连续出现5次0，认为任务已经完成 ========================================================================================================================
            if (self.category >= -0.1 and self.category <= 0.1):
                self.stop_count += 1
            else:
                self.stop_count = 0
            if self.stop_count >= 5:
                self.done = True

            if self.done == False:
                if (self.action >= -0.1 and self.category <= 0.1):
                    # 停住
                    None
                elif (self.action >= 0.1 and self.category <= 1.1):
                    # 直行25cm
                    self.msg.x = self.pos.x + 0.25 * math.cos(self.yaw)
                    self.msg.y = self.pos.y + 0.25 * math.sin(self.yaw)
                elif (self.action >= 1.9 and self.category <= 2.1):
                    # 左转30度
                    self.msg.theta = self.yaw + math.radians(30)
                elif (self.action >= 2.9 and self.category <= 3.1):
                    # 右转30度
                    self.msg.theta = self.yaw - math.radians(30)
                else:
                    # 停住
                    None
                self.waypoint_pub.publish(self.msg)     

            if self.done == True:
                # 第三类问题 ========================================================================================================================================
                if (self.category >= 2.9 and self.category <= 3.1):
                    self.subtask_idx += 1  
                    self.done = False
                    self.stop_count = 0

                # 第二类问题 ========================================================================================================================================
                if (self.category >= 1.9 and self.category <= 2.1):
                    # 用YOLO拉框，并通过话题发送（每个 subtask 完成后执行一次）
                    if self.obs["rgb"] is None:
                        rospy.logwarn("YOLO: obs['rgb'] is None, skip.")
                        return
                    # 加载YOLO
                    if self.yolo is None:
                        try:
                            self.yolo = YOLO('yolov8n.pt')
                            rospy.loginfo("YOLO model loaded.")
                        except Exception as e:
                            rospy.logerr("Failed to load YOLO model: %s", str(e))
                            return

                    try:
                        # YOLO 推理
                        bgr = self.obs["rgb"]
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        results = self.yolo.predict(source=rgb, verbose=False)

                        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                            rospy.logwarn("YOLO: no detections.")
                            return

                        res = results[0]
                        names = res.names  # id -> label 映射
                        cls_ids = res.boxes.cls.tolist()
                        labels = [names[int(c)] for c in cls_ids]
                        unique_labels = sorted(set(labels))

                        # 保留与keyeord最匹配的label（可能有多个）==========================================================================================================
                        selected_label = None
                        keyword = (self.keyword or "none").strip()
                        if ((self.category >= 0.9 and self.category <= 1.1) or (self.category >= 1.9 and self.category <= 2.1)) and len(unique_labels) > 1 and keyword.lower() != "none":
                            try:
                                os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
                                os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
                                client = OpenAI()
                                options_line = ", ".join(unique_labels)

                                messages = [
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a label selector. Your goal is to choose EXACTLY ONE label "
                                            "from the candidate list that best matches the given keyword.\n"
                                            "Strict rules:\n"
                                            "1) Always return EXACTLY one label FROM THE CANDIDATE LIST, spelled EXACTLY as given.\n"
                                            "2) Prefer the candidate that matches the SAME SEMANTIC TYPE as the keyword "
                                            "(e.g., lighting/fixture vs plant vs furniture vs appliance vs container, etc.).\n"
                                            "3) If multiple candidates share the same type, choose the one semantically closest to the keyword "
                                            "(synonyms, hyponyms/hypernyms accepted: 'refrigerator' ≈ 'fridge', 'bin' ≈ 'trash can').\n"
                                            "4) If none share the same type, choose the closest by meaning; break ties with string similarity.\n"
                                            "5) Output ONLY the chosen label. Do not explain, do not add quotes or extra text."
                                        )
                                    },

                                    {"role": "user", "content": "keyword: wall lamp\ncandidates: potted plant, traffic light\nChoose ONE from candidates."},
                                    {"role": "assistant", "content": "traffic light"},

                                    {"role": "user", "content": "keyword: refrigerator\ncandidates: fridge, cabinet\nChoose ONE from candidates."},
                                    {"role": "assistant", "content": "fridge"},

                                    {"role": "user", "content": "keyword: chair\ncandidates: sofa, computer, table\nChoose ONE from candidates."},
                                    {"role": "assistant", "content": "sofa"},

                                    {"role": "user", "content": "keyword: plant\ncandidates: potted plant, traffic light\nChoose ONE from candidates."},
                                    {"role": "assistant", "content": "potted plant"},

                                    {"role": "user", "content": "keyword: bin\ncandidates: trash can, vase\nChoose ONE from candidates."},
                                    {"role": "assistant", "content": "trash can"},

                                    {
                                        "role": "user",
                                        "content": f"keyword: {keyword}\ncandidates: {options_line}\nChoose ONE from candidates."
                                    },
                                ]

                                completion = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=10, messages=messages)
                                raw = (completion.choices[0].message.content or "").strip()
                                cand = raw.strip("'").strip('"')

                                if cand in unique_labels:
                                    selected_label = cand
                                else:
                                    # GPT返回不在候选中：用相似度兜底 ======================================================================================================
                                    best = difflib.get_close_matches(keyword.lower(), [u.lower() for u in unique_labels], n=1, cutoff=0.0)
                                    selected_label = (
                                        unique_labels[[u.lower() for u in unique_labels].index(best[0])]
                                        if best else unique_labels[0]
                                    )

                                rospy.loginfo(f"Label selection -> keyword='{keyword}', chosen='{selected_label}', candidates={unique_labels}")

                            except Exception as e:
                                rospy.logwarn(f"GPT label selection failed, fallback to heuristic: {e}")
                                best = difflib.get_close_matches(keyword.lower(), [u.lower() for u in unique_labels], n=1, cutoff=0.0)
                                selected_label = (
                                    unique_labels[[u.lower() for u in unique_labels].index(best[0])]
                                    if best else unique_labels[0]
                                )
                        else:
                            # 非1/2类或只有1个候选或keyword无效：直接用唯一/第一个
                            selected_label = unique_labels[0]

                        # 只保留选中标签中置信度最高的那个框 ================================================================================================================
                        conf_list = res.boxes.conf.tolist()
                        indices = [i for i, lbl in enumerate(labels) if lbl == selected_label]

                        if indices:
                            # 选置信度最大的那个索引
                            best_idx = max(indices, key=lambda i: conf_list[i] if i < len(conf_list) else -1.0)

                            # 只保留该 index 的框
                            keep_mask = [i == best_idx for i in range(len(labels))]
                            res.boxes = res.boxes[keep_mask]

                            # 日志里也只显示这个框
                            labels = [labels[best_idx]]

                        # 可视化并发布
                        annotated_bgr = res.plot()
                        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
                        annotated_msg.header.stamp = rospy.Time.now()
                        annotated_msg.header.frame_id = "map"
                        self.yolo_img_pub.publish(annotated_msg)

                        rospy.loginfo(f"YOLO published with label='{selected_label}', kept {len(labels)} boxes.")

                        # 假设只保留了一个 box
                        self.obj_box = res.boxes[0]
                        # xyxy 格式 [x1, y1, x2, y2]，是像素坐标
                        x1, y1, x2, y2 = self.obj_box.xyxy[0].tolist()
                        # 中心点（像素坐标）
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        # 映射比例
                        scale = 0.01
                        objMidX = cx * scale
                        objMidY = cy * scale
                        objMidZ = 0.0   # 图像是 2D 的，默认 0
                        # Marker 消息
                        self.marker.header.frame_id = "map"
                        self.marker.header.stamp = rospy.Time.now()
                        self.marker.ns = selected_label
                        self.marker.id = 0
                        self.marker.type = Marker.CUBE
                        self.marker.action = Marker.ADD
                        # 位置
                        self.marker.pose.position.x = objMidX
                        self.marker.pose.position.y = objMidY
                        self.marker.pose.position.z = objMidZ
                        # 发布 Marker
                        self.marker_pub.publish(self.marker)

                        rospy.loginfo(f"Published CUBE marker for {selected_label} at "
                                      f"({objMidX:.2f}, {objMidY:.2f}, {objMidZ:.2f}) "
                                      f"size=({self.marker.scale.x:.2f}, {self.marker.scale.y:.2f}, {self.marker.scale.z:.2f})")
                    except Exception as e:
                        rospy.logerr("YOLO inference failed: %s", str(e))


class NaVid_Agent(Agent):
    def __init__(self, model_path, result_path, require_map=True):
        
        print("Initialize NaVid")
        
        self.result_path = result_path
        self.require_map = require_map
        self.conv_mode = "vicuna_v1"

        # os.makedirs(self.result_path, exist_ok=True)
        # os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        # os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)


        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))


        print("Initialization Complete")

        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. \
                               You have been given a video of historical observations and an image of the current observation <image>. \
                               Your assigned task is: '{}'. \
                               Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance. \
                               Please note: what you see is only a partial scene directly in front of you, not the whole scene. \
                               Therefore, those areas beyond the field of view CANNOT BE IGNORED!"

        self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.reset()


    def process_images(self, rgb_list):
        
        start_img_index = 0
        
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        
        batch_image = np.asarray(rgb_list[start_img_index:])
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim = 0)
        

        return [self.history_rgb_tensor]


    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)


        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs



    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None



    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]



        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line


        if line:

            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)


        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image


    def reset(self):
                
        # if self.require_map:
        #     if len(self.topdown_map_list)!=0:
        #         output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))
        #         imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False
        


    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            
            return {"action": temp_action}


        navigation_qs = self.promt_template.format(observations["instruction"]["text"])
        navigation = self.predict_inference(navigation_qs)
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)


        action_index, num = self.extract_result(navigation[:-1])




        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num/25))):
                self.pending_action_list.append(1)

        elif action_index == 2:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(2)

        elif action_index == 3:
            for _ in range(min(3,int(num/30))):
                self.pending_action_list.append(3)
        
        if action_index is None or len(self.pending_action_list)==0:
            self.pending_action_list.append(random.randint(1, 3))
            # Primarily unused, intended to complete the pipeline logic.

        print("------------", action_index, " ---- ", num, " ---- ", len(self.rgb_list))

        
        return {"action": self.pending_action_list.pop(0)}

        
