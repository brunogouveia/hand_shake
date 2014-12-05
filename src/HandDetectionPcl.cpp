#include "ros/ros.h"
#include "boost/foreach.hpp"
// #include "std_msgs/String.h"

// #include <sensor_msgs/image_encodings.h>
// #include <sensor_msgs/CameraInfo.h>
// #include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <visualization_msgs/Marker.h>

#include <pcl_tools/pcl_utils.h>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include <pcl/ModelCoefficients.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/features/normal_3d.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/filters/extract_indices.h>
#include "nnn/nnn.hpp"

// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>

// #include <opencv/cv.h>
// #include "pinhole_camera_model.h"
// #include "depth_traits.h"

#include <iostream>
#include <sstream>

using namespace pcl;
using namespace std;

//find the points that are ajoining a cloud, but not in it:
//cloud: the full cloud
//cloudpts a vector of indices into cloud that represents the cluster for which we want to find near points
//centroid: the centroid of the nearby pts
//return: true if points were found within 5cm
bool findNearbyPts(pcl::PointCloud<pcl::PointXYZ> &cloud, std::vector<int> &cloudpts, Eigen::Vector4f &centroid){
	std::vector<int> inds(cloud.size(),1); //a way of marking the points we have looked at
	// 1: not in the cluster  0: in the cluster, seen  -1: in the cluster, not seen
	std::vector<int> nearpts; //a way of marking the points we have looked at
	std::vector<int> temp;
	for(uint i=0;i<cloudpts.size(); ++i) inds[cloudpts[i]]=-1;
	for(uint i=0;i<cloudpts.size(); ++i){
		if(inds[cloudpts[i]]==-1){
			NNN(cloud,cloud.points[cloudpts[i]],temp, .05);
			
			for(uint j=0;j<temp.size(); ++j){
				if(inds[temp[j]]==1){
					nearpts.push_back(temp[j]);
					inds[temp[j]]=2;
				}
				else
					inds[temp[j]]=-2;
			}
		}
	}
	//TODO: check if we are really just seeing the other hand:
	//		 remove any points that do not have a point w/in 1cm
	if(nearpts.size())
	//now find the centroid of the nearcloud:
		pcl::compute3DCentroid(cloud,nearpts,centroid);
	else
		return false;
	return true;
}

class PointCloudXyz {
private: 
	ros::Publisher pointCloudPub;
	ros::Publisher markerPub;

	ros::Subscriber pointCloudSub;

	
public:
	PointCloudXyz()	{}

	~PointCloudXyz() {}

	void init();

	void getSubCloud(PointCloud<PointXYZ> & cloudin, vector<int> & inds, PointCloud<PointXYZ> & cloudout) {
		pcl::ExtractIndices<PointXYZ> extract;
		// Extract the inliers
		extract.setInputCloud (cloudin.makeShared());
		extract.setIndices (boost::make_shared<vector<int> > (inds));
		extract.setNegative (false);
		extract.filter (cloudout);
	}

	void pubMarker(Eigen::Vector4f & point) {
		uint32_t shape = visualization_msgs::Marker::CUBE;

		visualization_msgs::Marker marker;
	    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
	    marker.header.frame_id = "/camera_depth_frame";
	    marker.header.stamp = ros::Time::now();

	    // Set the namespace and id for this marker.  This serves to create a unique ID
	    // Any marker sent with the same namespace and id will overwrite the old one
	    marker.ns = "basic_shapes";
	    marker.id = 0;

	    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
	    marker.type = shape;

	    // Set the marker action.  Options are ADD and DELETE
	    marker.action = visualization_msgs::Marker::ADD;

	    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
	    marker.pose.position.x = point(2);
	    marker.pose.position.y = -point(0);
	    marker.pose.position.z = -point(1);
	    marker.pose.orientation.x = 0.0;
	    marker.pose.orientation.y = 0.0;
	    marker.pose.orientation.z = 0.0;
	    marker.pose.orientation.w = 1.0;

	    // Set the scale of the marker -- 1x1x1 here means 1m on a side
	    marker.scale.x = .05;
	    marker.scale.y = .05;
	    marker.scale.z = .05;

	    // Set the color -- be sure to set alpha to something non-zero!
	    marker.color.r = 0.0f;
	    marker.color.g = 1.0f;
	    marker.color.b = 0.0f;
	    marker.color.a = 1.0;

	    marker.lifetime = ros::Duration();

	    markerPub.publish(marker);
	}

	void pclCallback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr ptr);
};

void PointCloudXyz::init()
{
	ros::NodeHandle nh;

	pointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/camera/rgb/points", 1);
	markerPub     = nh.advertise<visualization_msgs::Marker>("/marker", 1);
	pointCloudSub = nh.subscribe("/camera/depth/points", 1, &PointCloudXyz::pclCallback, this);
}


void PointCloudXyz::pclCallback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr ptr)
{
	PointCloud<PointXYZ> cloud = *ptr;
	vector<PointCloud<PointXYZ> > initialclouds;
	vector<Eigen::Vector4f> armCenter;
	std::vector<int> inds3(cloud.size(),1);

	//Find points near to camera
	PointXYZ camera(0,0,0);
	vector<int> inds;
	vector<float> distances;

	NNN(cloud, camera, inds, distances, 1.0f);
	
	int index=0; double smallestdist;
	for(uint i=0; i < distances.size(); ++i){
		if(distances[i] < smallestdist || i == 0 ){
			index = inds[i];
			smallestdist = distances[i];
		}
	}
	smallestdist = sqrt(smallestdist);
	PointXYZ closestPoint = cloud.points[index];

	NNN(cloud, closestPoint, inds, .2);


	//if there is nothing near that point, we're probably seeing noise.  just give up
	if(inds.size() < 100){
		std::cout<<"very few points " << endl;
		return;
	}

	Eigen::Vector4f centroid;
	PointXYZ centroidPoint;

	compute3DCentroid(cloud, inds, centroid);
	centroidPoint.x = centroid(0);
	centroidPoint.y = centroid(1)-.02;
	centroidPoint.z = centroid(2);

	NNN(cloud, centroidPoint, inds, .1);

	// //in the middle of everything, locate where the arms is:
	// std::vector<int> temp;
	// NNN(cloud,centroidPoint,temp, .15);
	// //finding the arms is really reliable. we'll just throw out anytime when we can't find it.
	// if(!findNearbyPts(cloud,temp,centroid))
	// 	return;

	// pcl::compute3DCentroid(cloud,inds,centroid);
	// centroidPoint.x=centroid(0); centroidPoint.y=centroid(1)-.01; centroidPoint.z=centroid(2);
	// NNN(cloud,centroidPoint,inds, .1);

	PointCloud<PointXYZ> cloudout;
	getSubCloud(cloud,inds,cloudout);
	// //-------Decide whether we are looking at potential hands:
	// //try to classify whether this is actually a hand, or just a random object (like a face)
	// //if there are many points at the same distance that we did not grab, then the object is not "out in front"
	// for(uint i=0;i<inds.size(); ++i) inds3[inds[i]]=0; //mark in inds3 all the points in the potential hand
	// pcl::compute3DCentroid(cloudout,centroid);
	// int s1,s2=0;
	// s1=inds.size();
	// //search for all points in the cloud that are as close as the center of the potential hand:
	// NNN(cloud,camera,inds, centroid.norm());
	// for(uint i=0;i<inds.size(); ++i){
	// 	if(inds3[inds[i]]) ++s2;
	// }
	// if(((float)s2)/((float)s1) > 25){
	// 	std::cout<<"No hands detected " << ((float)s2)/((float)s1)<< endl;
	// 	//return false; //uncomment
	// 	return;
	// }

	cout << centroid(0) << " " << centroid(1) << " " << centroid(2) << endl;

	pubMarker(centroid);
	pointCloudPub.publish(cloudout);
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
	 * way to do it.	The third argument to init() is the name of the node.
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
	// ros::NodeHandle n;

	/**
	 * The advertise() function is how you tell ROS that you want to
	 * publish on a given topic name. This invokes a call to the ROS
	 * master node, which keeps a registry of who is publishing and who
	 * is subscribing. After this advertise() call is made, the master
	 * node will notify anyone who is trying to subscribe to this topic name,
	 * and they will in turn negotiate a peer-to-peer connection with this
	 * node.	advertise() returns a Publisher object which allows you to
	 * publish messages on that topic through a call to publish().	Once
	 * all copies of the returned Publisher object are destroyed, the topic
	 * will be automatically unadvertised.
	 *
	 * The second parameter to advertise() is the size of the message queue
	 * used for publishing messages.	If messages are published more quickly
	 * than we can send them, the number here specifies how many messages to
	 * buffer up before throwing some away.
	 */
	// camera_info_pub = n.advertise<sensor_msgs::CameraInfo>("camera_info", 1);
	// image_pub = n.advertise<sensor_msgs::Image>("image_rect", 1);

	// ros::Subscriber camera_info_sub = n.subscribe("camera/depth/camera_info", 1, cameraCallback)
	// ros::Subscriber image_sub = n.subscribe("camera/depth/image", 1, imageCallback);

	 PointCloudXyz pt;
	 pt.init();

	ros::spin();

	return 0;
}