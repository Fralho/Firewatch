from i2c_hmc5883l import HMC5883
from time import sleep
import rospy
from std_msgs.msg import Float64
from storm32_controller import Storm32Controller

storm32 = Storm32Controller(port="/dev/ttyAMA4", baudrate=9600)
# storm32.set_angle_advanced(-5,0,-0)

def azimuth_direction(azimuth1, azimuth2):
    diff = (azimuth2 - azimuth1) % 360
    if diff <= 180:
        angle = diff
        signed_angle = angle
    else:
        angle = 360 - diff
        signed_angle = -angle
    
    return signed_angle

heading_mav = 0
def compass_callback(msg):
    global heading_mav
    heading_mav = msg.data
rospy.init_node('compass_reader')
rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, compass_callback)

i2c_HMC5883l = HMC5883(gauss=4)
#Set declination according to your position
i2c_HMC5883l.set_declination(12, 71)

for i in range(15):
   if i>5:
      heading_gim = i2c_HMC5883l.get_heading()[0]+90
      if heading_gim > 360: heading_gim-=360
      print(heading_gim, int(heading_mav), x:=azimuth_direction(heading_gim, int(255)))

      storm32.set_angle_advanced(-5,0,-x/1.5+storm32.get_current_angles()[2])
      print(storm32.get_current_angles()[2], i)
   sleep(.9)
