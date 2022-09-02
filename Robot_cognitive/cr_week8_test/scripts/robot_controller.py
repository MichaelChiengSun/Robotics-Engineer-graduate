#!/usr/bin/env python
import rospy

import imp; imp.find_module('bayesian_belief_networks')
from bayesian_belief_networks.ros_utils import *
from bayesian_belief_networks.msg import Observation, Result
from bayesian_belief_networks.srv import Query

from cr_week8_test.msg import perceived_info, robot_info
from cr_week8_test.srv import predict_robot_expression, predict_robot_expressionRequest, predict_robot_expressionResponse



id_filter = 0
object_size_filter = 0
human_expression_filter = 0
human_action_filter = 0

# Call back function is executed when message is received
def callback(data):
	global id_filter
	id_filter = data.id
	global object_size_filter
	object_size_filter = data.object_size
	global human_expression_filter
	human_expression_filter = data.human_expression
	global human_action_filter
	human_action_filter = data.human_action
	
	rospy.loginfo("Id:%i",id_filter)
	rospy.loginfo("object_size:%i",object_size_filter)
	rospy.loginfo("human_action:%i",human_action_filter)
	rospy.loginfo("human_expression:%i",human_expression_filter)

	obs = []
	if human_expression_filter != 0:
		obs.append(['human_expression', str(human_expression_filter)])
	if human_action_filter != 0:
		obs.append(['human_action', str(human_action_filter)])
	if object_size_filter != 0:
		obs.append(['object_size', str(object_size_filter)])
	ros_bbn_client(obs)

# Call remote ROS service
def ros_bbn_client(obss):
    rospy.wait_for_service('{}'.format('robot_expression_prediction/query'))
    try:
        query = rospy.ServiceProxy('robot_expression_prediction/query', Query)
        msg = []
        for obs in obss:
            o = Observation()
            o.node = obs[0]
            o.evidence = obs[1]
            msg.append(o)
        resp1 = query(msg)
	result = resp1.results[0:3]

	rospy.loginfo("Robot happy probability: %f",result[2].Marginal)
	rospy.loginfo("Robot sad probability: %f", result[0].Marginal)
	rospy.loginfo("Robot neutral probability: %f", result[1].Marginal)
	publish_robot_info(result)
        return resp1.results
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# Publish to a new ROS topic using robot_info message 
def publish_robot_info(res):
	prob_happy = res[2].Marginal
	prob_sad = res[0].Marginal
	prob_neutral = res[1].Marginal
	pub = rospy.Publisher('robotInfo', robot_info, queue_size=10)
	robot_expression = robot_info()
	robot_expression.id = id_filter
	robot_expression.p_happy = prob_happy
	robot_expression.p_sad = prob_sad
	robot_expression.p_neutral = prob_neutral
	pub.publish(robot_expression)

# Subscribe to node 2
def robot_controller():
	
	p = perceived_info()
	# Initialize node
	rospy.init_node('robot_controller')
	# Subscribe to topic from Node 2 (perception filter)
	rate = rospy.Rate(0.1)
	rospy.Subscriber('perceivedInfo', perceived_info, callback, queue_size=10)
	rate.sleep()
	rospy.spin()

if __name__ == '__main__':
	robot_controller()



	

