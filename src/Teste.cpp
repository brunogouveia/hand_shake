#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv/cv.h>
#include "pinhole_camera_model.h"
#include "depth_traits.h"

#include <iostream>
#include <sstream>

typedef sensor_msgs::PointCloud2 PointCloud;

namespace enc = sensor_msgs::image_encodings;

//Ros publishers
ros::Publisher pointCloudPub;
image_transport::Publisher newImagePub;

//Ros subscribers
ros::Subscriber imageSub;
ros::Subscriber cameraInfoSub;


//Camera model
image_geometry::PinholeCameraModel cameraModel;


template<typename T>
void convert(const sensor_msgs::ImageConstPtr& depth_msg, PointCloud::Ptr& cloud_msg)
{
	// Use correct principal point from calibration
	float center_x = cameraModel.cx();
	float center_y = cameraModel.cy();

	// Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
	double unit_scaling = DepthTraits<T>::toMeters( T(1) );
	float constant_x = unit_scaling / cameraModel.fx();
	float constant_y = unit_scaling / cameraModel.fy();
	float bad_point = std::numeric_limits<float>::quiet_NaN();

	sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
	sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
	sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
	const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
	int row_step = depth_msg->step / sizeof(T);
	for (int v = 0; v < (int)cloud_msg->height; ++v, depth_row += row_step)
	{
		for (int u = 0; u < (int)cloud_msg->width; ++u, ++iter_x, ++iter_y, ++iter_z)
		{
			T depth = depth_row[u];

			// Missing points denoted by NaNs
			if (!DepthTraits<T>::valid(depth))
			{
				*iter_x = *iter_y = *iter_z = bad_point;
				continue;
			}

			// Fill in XYZ
			*iter_x = (u - center_x) * depth * constant_x;
			*iter_y = (v - center_y) * depth * constant_y;
			*iter_z = DepthTraits<T>::toMeters(depth);
		}
	}
}

void depthCb(const sensor_msgs::ImageConstPtr& depth_msg)
{
	//Convert to opencv
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(depth_msg, enc::TYPE_32FC1);
	}
	catch (cv_bridge::Exception & e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	cv::imshow("Depth Image", cv_ptr->image);
	cv::waitKey(3);
		// std::cout << this << " " << &point_cloud_pub_ << std::endl;

	PointCloud::Ptr cloud_msg(new PointCloud);
	cloud_msg->header = depth_msg->header;
	cloud_msg->height = depth_msg->height;
	cloud_msg->width  = depth_msg->width;
	cloud_msg->is_dense = false;
	cloud_msg->is_bigendian = false;

	sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
	pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

	if (depth_msg->encoding == enc::TYPE_16UC1)
	{
		convert<uint16_t>(depth_msg, cloud_msg);
	}
	else if (depth_msg->encoding == enc::TYPE_32FC1)
	{
		convert<float>(depth_msg, cloud_msg);
	}
	else
	{
		ROS_ERROR("Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
		return;
	}


	pointCloudPub.publish (cloud_msg);
	newImagePub.publish(cv_ptr->toImageMsg());
}

void imageCallback(const sensor_msgs::ImageConstPtr& depth_msg) 
{
	std::cout << "imageCallback" <<std::endl;
	depthCb(depth_msg);
}

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& infoMsg) 
{
	cameraModel.fromCameraInfo(infoMsg);
}

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{
	/**
	 * The ros::init() function needs to see argc and argv so that it can perform
	 * any ROS arguments and name remapping that were provided at the command line. For programmatic
	 * remappings you can use a different version of init() which takes remappings
	 * directly, but for most command-line programs, passing argc and argv is the easiest
	 * way to do it.  The third argument to init() is the name of the node.
	 *
	 * You must call one of the versions of ros::init() before using any other
	 * part of the ROS system.
	 */
	ros::init(argc, argv, "HandDetection");

	/**
	 * NodeHandle is the main access point to communications with the ROS system.
	 * The first NodeHandle constructed will fully initialize this node, and the last
	 * NodeHandle destructed will close down the node.
	 */
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);

	/**
	 * The advertise() function is how you tell ROS that you want to
	 * publish on a given topic name. This invokes a call to the ROS
	 * master node, which keeps a registry of who is publishing and who
	 * is subscribing. After this advertise() call is made, the master
	 * node will notify anyone who is trying to subscribe to this topic name,
	 * and they will in turn negotiate a peer-to-peer connection with this
	 * node.  advertise() returns a Publisher object which allows you to
	 * publish messages on that topic through a call to publish().  Once
	 * all copies of the returned Publisher object are destroyed, the topic
	 * will be automatically unadvertised.
	 *
	 * The second parameter to advertise() is the size of the message queue
	 * used for publishing messages.  If messages are published more quickly
	 * than we can send them, the number here specifies how many messages to
	 * buffer up before throwing some away.
	 */
	// camera_info_pub = n.advertise<sensor_msgs::CameraInfo>("camera_info", 1);
	// image_pub = n.advertise<sensor_msgs::Image>("image_rect", 1);
	pointCloudPub = nh.advertise<PointCloud>("/points", 1);
	newImagePub   = it.advertise("/newImage", 1);
	imageSub      = nh.subscribe<sensor_msgs::Image>("camera/depth/image_raw", 1, imageCallback);
	cameraInfoSub = nh.subscribe<sensor_msgs::CameraInfo>("camera/depth/camera_info", 1, cameraInfoCallback);

	// ros::Subscriber camera_info_sub = n.subscribe("camera/depth/camera_info", 1, cameraCallback)
; // ros::Subscriber image_sub = n.subscribe("camera/depth/image", 1, imageCallback);


	ros::spin();

	return 0;
}