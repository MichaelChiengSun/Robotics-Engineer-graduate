#!/usr/bin/env python
import rospy
import random
from cr_week8_test.msg import object_info, human_info

# Frequency of publishing data
pubFreq = 0.1	# 10seconds = 0.1Hz

# Create object for both msg
O = object_info()
O.id = 0	# Initialize id
H = human_info()



def interaction_generator():
	# Tell rospy name of node
    	rospy.init_node('interaction_generator', anonymous=True)
    	# Publish to "object" and "human" topic
    	pub1 = rospy.Publisher('objectInfo', object_info, queue_size=10)
	pub2 = rospy.Publisher('humanInfo', human_info, queue_size=10)

    	rate = rospy.Rate(pubFreq)	# Publish at pubFreq Hz

    	while not rospy.is_shutdown():
		O.id += 1
		H.id = O.id
		O.object_size = random.choice([1,2])
		H.human_expression = random.choice([1,2,3])
		H.human_action = random.choice([1,2,3])

        	pub1.publish(O)
		pub2.publish(H)
		# Print message to screen
        	rospy.loginfo("ID:%i",H.id)
		rospy.loginfo("Object size:%i",O.object_size)
		rospy.loginfo("Human expression:%i",H.human_expression)
		rospy.loginfo("Human action:%i",H.human_action)

        	rate.sleep()	# Sleep for about 1/pubFreq seconds


# Main function
if __name__ == '__main__':
	try:
		interaction_generator()
	except rospy.ROSInterruptException:
		pass

