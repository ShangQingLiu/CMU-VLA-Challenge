import subprocess
import os
import time

if __name__ == "__main__":
    team_name = "ReasonX"

    subprocess.run(["xhost", "+"])
    compose_dir = f"/dataSSD/1sliu/TAMSxGalbot/CMU-VLA-Challenge/docker"
    
    subprocess.run(
        ["docker", "compose", "-f", "compose_gpu.yml", "up", "--build", "-d"],
        check=True,
        cwd=compose_dir
    )

    time.sleep(5)

    proc1 = subprocess.Popen(
        ["docker", "exec", "ubuntu20_ros_system",
"bash", "-c", "source /opt/ros/noetic/setup.bash && ./launch_system.sh"],
        preexec_fn=os.setsid
    )
    time.sleep(10)
    
    # --- modified PROC2: run a new container and execute launch.sh ---
    user = os.environ.get("USER", "user")
    display = os.environ.get("DISPLAY", ":0")
    image_id = "015e33d1bd60"
    
    cmd = (
    "source /opt/ros/noetic/setup.bash && "
    "./launch_module.sh 2>&1 | tee -a /tmp/launch_module.log"
)
    
    proc2 = subprocess.Popen(
        ["docker", "exec", "ubuntu20_ros",
"bash", "-lc", cmd],
        preexec_fn=os.setsid
    )

    cmd = (
        "source /opt/ros/noetic/setup.bash && "
        "rostopic pub -r 1 /challenge_question std_msgs/String "
        "\"data: 'How many red pillows are on the sofa?'\""
    )

    proc3 = subprocess.Popen(
        ["docker", "exec", "-i", "ubuntu20_ros", "bash", "-lc", cmd],
        preexec_fn=os.setsid
    )
