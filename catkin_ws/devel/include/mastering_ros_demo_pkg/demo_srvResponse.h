// Generated by gencpp from file mastering_ros_demo_pkg/demo_srvResponse.msg
// DO NOT EDIT!


#ifndef MASTERING_ROS_DEMO_PKG_MESSAGE_DEMO_SRVRESPONSE_H
#define MASTERING_ROS_DEMO_PKG_MESSAGE_DEMO_SRVRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace mastering_ros_demo_pkg
{
template <class ContainerAllocator>
struct demo_srvResponse_
{
  typedef demo_srvResponse_<ContainerAllocator> Type;

  demo_srvResponse_()
    {
    }
  demo_srvResponse_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> const> ConstPtr;

}; // struct demo_srvResponse_

typedef ::mastering_ros_demo_pkg::demo_srvResponse_<std::allocator<void> > demo_srvResponse;

typedef boost::shared_ptr< ::mastering_ros_demo_pkg::demo_srvResponse > demo_srvResponsePtr;
typedef boost::shared_ptr< ::mastering_ros_demo_pkg::demo_srvResponse const> demo_srvResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace mastering_ros_demo_pkg

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsMessage': True, 'IsFixedSize': True, 'HasHeader': False}
// {'mastering_ros_demo_pkg': ['/home/tan/catkin_ws/src/mastering_ros_demo_pkg/msg'], 'std_msgs': ['/opt/ros/lunar/share/std_msgs/cmake/../msg'], 'actionlib_msgs': ['/opt/ros/lunar/share/actionlib_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsMessage< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "mastering_ros_demo_pkg/demo_srvResponse";
  }

  static const char* value(const ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
";
  }

  static const char* value(const ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct demo_srvResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::mastering_ros_demo_pkg::demo_srvResponse_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // MASTERING_ROS_DEMO_PKG_MESSAGE_DEMO_SRVRESPONSE_H
