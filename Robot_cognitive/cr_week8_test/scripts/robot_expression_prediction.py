#!/usr/bin/env python
from cr_week8_test.srv import predict_robot_expression, predict_robot_expressionRequest, predict_robot_expressionResponse
import rospy
import imp; imp.find_module('bayesian_belief_networks')
from bayesian_belief_networks.ros_utils import *


def fHE(human_expression):
	'''Human_expression'''
	if human_expression == '1' or human_expression == '2' or human_expression == '3':
		return 1.0/3

def fHA(human_action):
	'''Human_action'''
	if human_action == '1' or human_action == '2' or human_action == '3':
		return 1.0/3
	
def fO(object_size):
	'''Object_size'''
	if object_size == '1' or object_size =='2':
		return 0.5

def fRE(human_expression,human_action, object_size, PRE):
        '''Robot happy'''
        table=dict()
        table['HRSH'] = 0.8
        table['HRBH'] = 1.0
        table['HOSH'] = 0.8
        table['HOBH'] = 0.1
        table['HASH'] = 0.6
        table['HABH'] = 0.8
        table['SRSH'] = 0.0
        table['SRBH'] = 0.0
        table['SOSH'] = 0.0
        table['SOBH'] = 0.1
        table['SASH'] = 0.0
        table['SABH'] = 0.2
        table['NRSH'] = 0.7
        table['NRBH'] = 0.8
        table['NOSH'] = 0.8
        table['NOBH'] = 0.9
        table['NASH'] = 0.6
        table['NABH'] = 0.7

	'''Robot sad'''
        table['HRSS'] = 0.2
        table['HRBS'] = 0.0
        table['HOSS'] = 0.2
        table['HOBS'] = 0.0
        table['HASS'] = 0.2
        table['HABS'] = 0.2
        table['SRSS'] = 0.0
        table['SRBS'] = 0.0
        table['SOSS'] = 0.1
        table['SOBS'] = 0.1
        table['SASS'] = 0.2
        table['SABS'] = 0.2
        table['NRSS'] = 0.3
        table['NRBS'] = 0.2
        table['NOSS'] = 0.2
        table['NOBS'] = 0.1
        table['NASS'] = 0.2
        table['NABS'] = 0.2

        '''Robot neutral'''
        table['HRSN'] = 0.0
        table['HRBN'] = 0.0
        table['HOSN'] = 0.0
        table['HOBN'] = 0.0
        table['HASN'] = 0.2
        table['HABN'] = 0.0
        table['SRSN'] = 1.0
        table['SRBN'] = 1.0
        table['SOSN'] = 0.9
        table['SOBN'] = 0.8
        table['SASN'] = 0.8
        table['SABN'] = 0.6
        table['NRSN'] = 0.0
        table['NRBN'] = 0.0
        table['NOSN'] = 0.0
        table['NOBN'] = 0.0
        table['NASN'] = 0.2
        table['NABN'] = 0.1

    	key = ''
	if human_expression == '1':
		key = key + 'H'
	if human_expression == '2':
		key = key + 'S'
	if human_expression == '3':
		key = key + 'N'

	if human_action == '1':
		key = key + 'R'
	if human_action == '2':
		key = key + 'O'
	if human_action == '3':
		key = key + 'A'

	if object_size == '1':
		key = key + 'S'
	if object_size == '2':
		key = key + 'B'

	if PRE == '1':
		key = key + 'H'
	if PRE == '2':
		key = key + 'S'
	if PRE == '3':
		key = key + 'N'

	return table[key]


if __name__ == "__main__":
	rospy.init_node('robot_expression_prediction')
	g = ros_build_bbn(fHE, fHA, fO, fRE,
		domains={
			'human_expression': ['1', '2', '3'],
			'human_action': ['1', '2', '3'],
			'object_size': ['1', '2'],
			'PRE': ['1', '2', '3']}) 

	rospy.spin()



