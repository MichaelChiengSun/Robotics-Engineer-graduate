#!/usr/bin/env python
import rospy
import random
from cr_week8_test.msg import object_info, human_info, perceived_info


identification = 0
object_size = 0
human_expression = 0
human_action = 0

def callback_object_info(obj_data):
	global object_size
	object_size = obj_data.object_size

def callback_human_info(hum_data):
	global identification
	identification = hum_data.id
	global human_expression
	human_expression = hum_data.human_expression
	global human_action
	human_action = hum_data.human_action

def perception_filter():
    	# Tell rospy name of node
    	rospy.init_node('perception_filter', anonymous=True)
    	# Subscribe to both topics from Node 1 (interaction_generator)
    	object_sub = rospy.Subscriber('objectInfo', object_info, callback_object_info, queue_size=10)
    	human_sub = rospy.Subscriber('humanInfo', human_info, callback_human_info, queue_size=10)
	

if __name__ == '__main__':
    	perception_filter()
	# Create a new publisher topic for filtered information
	pub = rospy.Publisher('perceivedInfo', perceived_info, queue_size=10)
	rate = rospy.Rate(0.1)
	filtered_data = perceived_info()
	while not rospy.is_shutdown():
		filtered_data.id = identification
		filtered_data.object_size = object_size
		filtered_data.human_expression = human_expression
		filtered_data.human_action = human_action
		random_filter = random.randint(1,8)
		if random_filter == 1:
			filtered_data.object_size = 0
		if random_filter == 2:
			filtered_data.human_action = 0
		if random_filter == 3:
			filtered_data.human_expression = 0
		if random_filter == 4:
			filtered_data.object_size = 0
			filtered_data.human_action = 0
		if random_filter == 5:
			filtered_data.object_size = 0
			filtered_data.human_expression = 0
		if random_filter == 6:
			filtered_data.human_action = 0
			filtered_data.human_expression = 0
		if random_filter == 7:
			filtered_data.object_size = 0
			filtered_data.human_action = 0
			filtered_data.human_expression = 0
		# Publish filtered information to topic
		pub.publish(filtered_data)
		# Display results
		rospy.loginfo("Id:%i",filtered_data.id)
		rospy.loginfo("Object size:%i",filtered_data.object_size)
		rospy.loginfo("Human expression:%i",filtered_data.human_expression)
		rospy.loginfo("Human action:%i",filtered_data.human_action)
		
		rate.sleep()

	
		
