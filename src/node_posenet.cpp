/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ros_compat.h"
#include "image_converter.h"

#include <jetson-inference/poseNet.h>

#include <unordered_map>


// globals
poseNet* net = NULL;
uint32_t overlay_flags = poseNet::OVERLAY_DEFAULT;

imageConverter* input_cvt   = NULL;
imageConverter* overlay_cvt = NULL;

Publisher<sensor_msgs::Image> overlay_pub = NULL;
Publisher<vision_msgs::VisionInfo> info_pub = NULL;

vision_msgs::VisionInfo info_msg;


// triggered when a new subscriber connected
void info_callback()
{
	ROS_INFO("new subscriber connected to vision_info topic, sending VisionInfo msg");
	info_pub->publish(info_msg);
}


// publish overlay image
bool publish_overlay( )
{
	// get the image dimensions
	const uint32_t width  = input_cvt->GetWidth();
	const uint32_t height = input_cvt->GetHeight();

	// assure correct image size
	//if( !overlay_cvt->Resize(width, height, imageConverter::ROSOutputFormat) )
	//	return false;

	// run pose estimation
	std::vector<poseNet::ObjectPose> pose;

	if( !net->Process(input_cvt->ImageGPU(), width, height, pose, overlay_flags))
	{		
		LogError("posenet: faild to process frame\n");
	}
	LogInfo("posenet: detected %zu %s(s)\n", pose.size(), net->GetCategory());

	// populate the message
	sensor_msgs::Image msg;

	if( !input_cvt->Convert(msg, imageConverter::ROSOutputFormat) )
		return false;

	// populate timestamp in header field
	msg.header.stamp = ROS_TIME_NOW();

	// publish the message	
	overlay_pub->publish(msg);
	ROS_DEBUG("publishing %ux%u overlay image", width, height);
}


// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr input )
{
	// convert the image to reside on GPU
	if( !input_cvt || !input_cvt->Convert(input) )
	{
		ROS_INFO("failed to convert %ux%u %s image", input->width, input->height, input->encoding.c_str());
		return;	
	}
	if( ROS_NUM_SUBSCRIBERS(overlay_pub) > 0 )
		publish_overlay();
}


// node main loop
int main(int argc, char **argv)
{
	/*
	 * create node instance
	 */
	ROS_CREATE_NODE("posenet");

	/*
	 * retrieve parameters
	 */	
	std::string model_name  = "resnet18-body";
	std::string model_path;
	std::string prototxt_path;
	std::string overlay_str = "OVERLAY_DEFAULT";	
	std::string input_blob  = POSENET_DEFAULT_INPUT;
	std::string output_cmap  = POSENET_DEFAULT_CMAP;
	std::string output_paf = POSENET_DEFAULT_PAF;
	float keypoint_scale = POSENET_DEFAULT_KEYPOINT_SCALE;
	float link_scale = POSENET_DEFAULT_LINK_SCALE;

	float mean_pixel = 0.0f;
	float threshold  = POSENET_DEFAULT_THRESHOLD;

	ROS_DECLARE_PARAMETER("model_name", model_name);
	ROS_DECLARE_PARAMETER("model_path", model_path);
	ROS_DECLARE_PARAMETER("prototxt_path", prototxt_path);
	ROS_DECLARE_PARAMETER("input_blob", input_blob);
	ROS_DECLARE_PARAMETER("output_cmap", output_cmap);
	ROS_DECLARE_PARAMETER("output_paf", output_paf);
	ROS_DECLARE_PARAMETER("keypoint_scale", keypoint_scale);
	ROS_DECLARE_PARAMETER("link_scale", link_scale);
	ROS_DECLARE_PARAMETER("threshold", threshold);
	ROS_DECLARE_PARAMETER("mean_pixel", mean_pixel);
	ROS_DECLARE_PARAMETER("overlay_flags", overlay_flags);

	/*
	 * retrieve parameters
	 */
	ROS_GET_PARAMETER("model_name", model_name);
	ROS_GET_PARAMETER("model_path", model_path);
	ROS_GET_PARAMETER("prototxt_path", prototxt_path);
	ROS_GET_PARAMETER("input_blob", input_blob);
	ROS_GET_PARAMETER("output_cmap", output_cmap);
	ROS_GET_PARAMETER("output_paf", output_paf);
	ROS_GET_PARAMETER("keypoint_scale", keypoint_scale);
	ROS_GET_PARAMETER("link_scale", link_scale );
	ROS_GET_PARAMETER("threshold", threshold);
	ROS_GET_PARAMETER("mean_pixel", mean_pixel);
	ROS_GET_PARAMETER("overlay_flags", overlay_flags);

	/*
	 * load object detection network
	 */
	if( model_path.size() > 0 )
	{
		// create network using custom model paths
		net = poseNet::Create(model_name.c_str(), threshold, DEFAULT_MAX_BATCH_SIZE, TYPE_INT8, DEVICE_GPU, true );
	}
	else
	{
		// create network using the built-in model
		net = poseNet::Create(model_name.c_str());
	}

	if( !net )
	{
		ROS_ERROR("failed to load detectNet model");
		return 0;
	}


	

	/*
	 * create image converter objects
	 */
	input_cvt = new imageConverter();
	overlay_cvt = new imageConverter();

	if( !input_cvt || !overlay_cvt )
	{
		ROS_ERROR("failed to create imageConverter objects");
		return 0;
	}


	/*
	 * advertise publisher topics
	 */
	ROS_CREATE_PUBLISHER(sensor_msgs::Image, "overlay", 2, overlay_pub);
	
	ROS_CREATE_PUBLISHER_STATUS(vision_msgs::VisionInfo, "vision_info", 1, info_callback, info_pub);


	/*
	 * subscribe to image topic
	 */
	auto img_sub = ROS_CREATE_SUBSCRIBER(sensor_msgs::Image, "image_in", 5, img_callback);

	
	/*
	 * wait for messages
	 */
	ROS_INFO("posenet node initialized, waiting for messages");
	ROS_SPIN();


	/*
	 * free resources
	 */
	delete net;
	delete input_cvt;
	delete overlay_cvt;

	return 0;
}

