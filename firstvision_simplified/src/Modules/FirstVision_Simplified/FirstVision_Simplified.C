// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <jevoisbase/Components/Tracking/Kalman1D.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Geometry> // for AngleAxis and Quaternion

// For IPC
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>

using namespace boost::interprocess;

// Structure of IPC mapped memory
typedef struct
{
  double translation[3]; // Vector to the detected object
  double rotation[3];    // Orientation vector of the detected object
  long int proc_time;    // Time take to process this frame
} cam_ipc_data_t;

// REMINDER: make sure you understand the viral nature and terms of the above license. If you are writing code derived
// from this file, you must offer your source under the GPL license too.

static jevois::ParameterCategory const ParamCateg("FirstVision_Simplified Options");

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(hmin, unsigned char, "Initial min target hue (0=red/do not use because of "
                                                            "wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan, "
                                                            "105=light blue, 120=blue, 135=purple, 150=pink)",
                                       20, jevois::Range<unsigned char>(0, 179), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(hmax, unsigned char, "Initial max target hue (0=red/do not use because of "
                                                            "wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan, "
                                                            "105=light blue, 120=blue, 135=purple, 150=pink)",
                                       130, jevois::Range<unsigned char>(0, 179), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(houter, bool, "Use exclusive hue range",
                                       true, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(smin, unsigned char, "Set the sat min. Sat range is <smin, smax>.",
                                       40, jevois::Range<unsigned char>(0, 255), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(smax, unsigned char, "Set the sat max. Sat range is <smin, smax>.",
                                       255, jevois::Range<unsigned char>(0, 255), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(vmin, unsigned char, "Set the value min. Value range is <vmin, vmax>.",
                                       45, jevois::Range<unsigned char>(0, 255), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(vmax, unsigned char, "Set the value max. Value range is <vmin, vmax>.",
                                       255, jevois::Range<unsigned char>(0, 255), ParamCateg);

// //! Parameter \relates FirstVision_Simplified
// JEVOIS_DECLARE_PARAMETER(maxnumobj, size_t, "Max number of objects to declare a clean image. If more blobs are "
// 			 "detected in a frame, we skip that frame before we even try to analyze shapes of the blobs",
//                          100, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(fillratio_min, int, "Min fill ratio of the convex hull (percent). Higher values mean your shape "
                                             "occupies a higher fraction of its convex hull. This parameter sets a lower bound, "
                                             "less full shapes will be rejected.",
                         10, jevois::Range<int>(1, 100), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(fillratio_max, int, "Max fill ratio of the convex hull (percent). Lower values mean your shape "
                                             "occupies a smaller fraction of its convex hull. This parameter sets an upper bound, "
                                             "fuller shapes will be rejected.",
                         40, jevois::Range<int>(1, 100), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(erodesize, size_t, "Erosion structuring element size (pixels), or 0 for no erosion",
                         2, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(dilatesize, size_t, "Dilation structuring element size (pixels), or 0 for no dilation",
                         4, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(blursize, size_t, "Gaussian blur structuring element size (pixels), must be odd",
                         3, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(blursigma, size_t, "Sigma for gaussian blur",
                         1, ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(epsilon, double, "Shape smoothing factor (higher for smoother). Shape smoothing is applied "
                                          "to remove small contour defects before the shape is analyzed.",
                         0.015, jevois::Range<double>(0.001, 0.999), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(shapeerror_max, double, "Shape error threshold (lower is stricter for exact shape)",
                         900.0, jevois::Range<double>(0.01, 1000.0), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(objsize, cv::Size_<float>, "Object size (in meters)",
                         cv::Size_<float>(0.28F, 0.175F), ParamCateg);

//! Parameter \relates FirstVision_Simplified
JEVOIS_DECLARE_PARAMETER(camparams, std::string, "File stem of camera parameters, or empty. Camera resolution "
                                                 "will be appended, as well as a .yaml extension. For example, specifying 'calibration' "
                                                 "here and running the camera sensor at 320x240 will attempt to load "
                                                 "calibration320x240.yaml from within directory " JEVOIS_SHARE_PATH "/camera/",
                         "calibration", ParamCateg);

//! Simple color-based detection of a U-shaped object for FIRST Robotics
/*! This module isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some
    cleanups, and extracts object contours. It is looking for a rectangular U shape of a specific size (set by parameter
    \p objsize). See screenshots for an example of shape. It sends information about detected objects over serial.

    This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
    so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
    how to set the camera settings each time this module is loaded.

    This code was loosely inspired by the JeVois \jvmod{ObjectTracker} module. Also see \jvmod{FirstPython} for a
    simplified version of this module, written in Python.

    This module is provided for inspiration. It has no pretension of actually solving the FIRST Robotics vision problem
    in a complete and reliable way. It is released in the hope that FRC teams will try it out and get inspired to
    develop something much better for their own robot.

    General pipeline
    ----------------

    The basic idea of this module is the classic FIRST robotics vision pipeline: first, select a range of pixels in HSV
    color pixel space likely to include the object. Then, detect contours of all blobs in range. Then apply some tests
    on the shape of the detected blobs, their size, fill ratio (ratio of object area compared to its convex hull's
    area), etc. Finally, estimate the location and pose of the object in the world.

    In this module, we run up to 4 pipelines in parallel, using different settings for the range of HSV pixels
    considered:

    - Pipeline 0 uses the HSV values provided by user parameters;
    - Pipeline 1 broadens that fixed range a bit;
    - Pipelines 2-3 use a narrow and broader learned HSV window over time.

    Detections from all 4 pipelines are considered for overlap and quality (raggedness of their outlines), and only the
    cleanest of several overlapping detections is preserved. From those cleanest detections, pipelines 2-3 learn and
    adapt the HSV range for future video frames.

    Using this module
    -----------------

    Check out [this tutorial](http://jevois.org/tutorials/UserFirstVision.html).

    Detection and quality control steps
    -----------------------------------

    The following messages appear for each of the 4 pipelines, at the bottom of the demo video, to help users figure out
    why their object may not be detected:
    
    - T0 to T3: thread (pipeline) number
    - H=..., S=..., V=...: HSV range considered by that thread
    - N=...: number of raw blobs detected in that range
    - Because N blobs are considered in each thread from this point on, information about only the one that progressed
      the farthest through a series of tests is shown. One letter is added each time a test is passed:
      + H: the convex hull of the blob is quadrilateral (4 vertices)
      + A: hull area is within range specified by parameter \p hullarea
      + F: object to hull fill ratio is below the limit set by parameter \p hullfill (i.e., object is not a solid,
        filled quadrilateral shape)
      + S: the object has 8 vertices after shape smoothing to eliminate small shape defects (a U shape is
        indeed expected to have 8 vertices).
      + E: The shape discrepency between the original shape and the smoothed shape is acceptable per parameter 
        \p ethresh, i.e., the original contour did not have a lot of defects.
      + M: the shape is not too close to the borders of the image, per parameter \p margin, i.e., it is unlikely to 
        be truncated as the object partially exits the camera's field of view.
      + V: Vectors describing the shape as it related to its convex hull are non-zero, i.e., the centroid of the shape
        is not exactly coincident with the centroid of its convex hull, as we would expect for a U shape.
      + U: the shape is roughly upright; upside-down U shapes are rejected as likely spurious.
      + OK: this thread detected at least one shape that passed all the tests.

    The black and white picture at right shows the pixels in HSV range for the thread determined by parameter \p
    showthread (with value 0 by default).

    Serial Messages
    ---------------
 
    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued on
    every video frame for each detected and good object. The \p id field in the messages simply is \b FIRST for all
    messages.

    When \p dopose is turned on, 3D messages will be sent, otherwise 2D messages.

    2D messages when \p dopose is off:

    - Serial message type: \b 2D
    - `id`: always `FIRST`
    - `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string)

    3D messages when \p dopose is on:

    - Serial message type: \b 3D
    - `id`: always `FIRST`
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of object center, or corners
    - `w`, `h`, `d`: object size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string)

    NOTE: 3D pose estimation from low-resolution 176x144 images at 120fps can be quite noisy. Make sure you tune your
    HSV ranges very well if you want to operate at 120fps (see below). To operate more reliably at very low resolutions,
    one may want to improve this module by adding subpixel shape refinement and tracking across frames.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    Trying it out
    -------------

    The default parameter settings (which are set in \b script.cfg explained below) attempt to detect yellow-green
    objects. Present an object to the JeVois camera and see whether it is detected. When detected and good
    enough according to a number of quality control tests, the outline of the object is drawn.

    For further use of this module, you may want to check out the following tutorials:

    - [Using the sample FIRST Robotics vision module](http://jevois.org/tutorials/UserFirstVision.html)
    - [Tuning the color-based object tracker using a python graphical
      interface](http://jevois.org/tutorials/UserColorTracking.html)
    - [Making a motorized pan-tilt head for JeVois and tracking
      objects](http://jevois.org/tutorials/UserPanTilt.html)
    - \ref ArduinoTutorial

    Tuning
    ------

    You need to provide the exact width and height of your physical shape to parameter \p objsize for this module to
    work. It will look for a shape of that physical size (though at any distance and orientation from the camera). Be
    sure you edit \b script.cfg and set the parameter \p objsize in there to the true measured physical size of your
    shape.

    You should adjust parameters \p hcue, \p scue, and \p vcue to isolate the range of Hue, Saturation, and Value
    (respectively) that correspond to the objects you want to detect. Note that there is a \b script.cfg file in this
    module's directory that provides a range tuned to a light yellow-green object, as shown in the demo screenshot.

    Tuning the parameters is best done interactively by connecting to your JeVois camera while it is looking at some
    object of the desired color. Once you have achieved a tuning, you may want to set the hcue, scue, and vcue
    parameters in your \b script.cfg file for this module on the microSD card (see below).

    Typically, you would start by narrowing down on the hue, then the value, and finally the saturation. Make sure you
    also move your camera around and show it typical background clutter so check for false positives (detections of
    things which you are not interested, which can happen if your ranges are too wide).

    Config file
    -----------

    JeVois allows you to store parameter settings and commands in a file named \b script.cfg stored in the directory of
    a module. The file \b script.cfg may contain any sequence of commands as you would type them interactively in the
    JeVois command-line interface. For the \jvmod{FirstVision} module, a default script is provided that sets the camera
    to manual color, gain, and exposure mode (for more reliable color values), and other example parameter values.

    The \b script.cfg file for \jvmod{FirstVision} is stored on your microSD at
    <b>JEVOIS:/modules/JeVois/FirstVision/script.cfg</b> 

    @author Laurent Itti

    @videomapping YUYV 176 194 120.0 YUYV 176 144 120.0 JeVois FirstVision_Simplified
    @videomapping YUYV 352 194 120.0 YUYV 176 144 120.0 JeVois FirstVision_Simplified
    @videomapping YUYV 320 290 60.0 YUYV 320 240 60.0 JeVois FirstVision_Simplified
    @videomapping YUYV 640 290 60.0 YUYV 320 240 60.0 JeVois FirstVision_Simplified
    @videomapping NONE 0 0 0.0 YUYV 320 240 60.0 JeVois FirstVision_Simplified
    @videomapping NONE 0 0 0.0 YUYV 176 144 120.0 JeVois FirstVision_Simplified
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class FirstVision_Simplified : public jevois::StdModule,
                               public jevois::Parameter<hmin, hmax, houter, smin, smax, vmin, vmax, fillratio_max, fillratio_min, erodesize,
                                                        dilatesize, epsilon, shapeerror_max, objsize, camparams, blursize, blursigma>
{
protected:
  cv::Mat itsCamMatrix;      //!< Our camera matrix
  cv::Mat itsDistCoeffs;     //!< Our camera distortion coefficients
  bool itsCueChanged = true; //!< True when users change ranges

  void onParamChange(hmin const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(hmax const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(smin const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(smax const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(vmin const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(vmax const &param, unsigned char const &newval) { itsCueChanged = true; }
  void onParamChange(houter const &param, bool const &newval) { itsCueChanged = true; }

  // ####################################################################################################
  //! Helper struct for an HSV range triplet, where each range is specified as a min and max:
  /*! if outer is set to true, then  */
  struct hsvcue
  {
    //! Constructor
    hsvcue(unsigned char _h_min, unsigned char _h_max, unsigned char _s_min, unsigned char _s_max,
           unsigned char _v_min, unsigned char _v_max, bool _h_outer) : h_min(_h_min), h_max(_h_max), s_min(_s_min),
                                                                        s_max(_s_max), v_min(_v_min), v_max(_v_max), h_outer(_h_outer)
    {
      fix();
    }

    //! Fix ranges so they don't go out of bounds
    void fix()
    {
      h_min = std::min(179, std::max(0, h_min));
      h_max = std::min(179, std::max(0, h_max));
      s_min = std::min(255, std::max(0, s_min));
      s_max = std::min(255, std::max(0, s_max));
      v_min = std::min(255, std::max(0, v_min));
      v_max = std::min(255, std::max(0, v_max));

      if (h_min > h_max)
      {
        int tmp = h_min;
        h_min = h_max;
        h_max = tmp;
      }
      if (s_min > s_max)
      {
        int tmp = s_min;
        s_min = s_max;
        s_max = tmp;
      }
      if (v_min > v_max)
      {
        int tmp = v_min;
        v_min = v_max;
        v_max = tmp;
      }
    }

    //! Get minimum triplet for use by cv::inRange()
    cv::Scalar rmin() const
    {
      return cv::Scalar(h_min, s_min, v_min);
    }

    //! Get maximum triplet for use by cv::inRange()
    cv::Scalar rmax() const
    {
      return cv::Scalar(h_max, s_max, v_max);
    }

    int h_min, h_max; //!< Min and max for H
    int s_min, s_max; //!< Min and max for S
    int v_min, v_max; //!< Min and max for V
    bool h_outer;     //!< Exclusive range for H
  };

  std::vector<hsvcue> itsHSV;

  // ####################################################################################################
  //! Helper struct for a detected object
  struct detection
  {
    std::vector<cv::Point> contour; //!< The full detailed contour
    std::vector<cv::Point> approx;  //!< Smoothed approximation of the contour
    std::vector<cv::Point> hull;    //!< Convex hull of the contour
    float serr;                     //!< Shape error score (higher for rougher contours with defects)
    float fill_ratio;               //!< Area fill ratio
  };

  //! Our detections, combined across all threads
  std::vector<detection> itsDetections;
  std::mutex itsDetMtx;

  // ####################################################################################################

  //! Erosion and dilation kernels shared across all detect threads
  cv::Mat itsErodeElement, itsDilateElement;

  // Delete

  // ####################################################################################################
  //! ParallelLoopBody class for the parallelization of the single markers pose estimation
  /*! Derived from opencv_contrib ArUco module, it's just a simple solvePnP inside. */
  class SinglePoseEstimationParallel : public cv::ParallelLoopBody
  {
  public:
    SinglePoseEstimationParallel(cv::Mat &_objPoints, cv::InputArrayOfArrays _corners,
                                 cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
                                 cv::Mat &_rvecs, cv::Mat &_tvecs) : objPoints(_objPoints), corners(_corners), cameraMatrix(_cameraMatrix),
                                                                     distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs) {}

    void operator()(cv::Range const &range) const
    {
      int const begin = range.start;
      int const end = range.end;

      for (int i = begin; i < end; ++i)
        cv::solvePnP(objPoints, corners.getMat(i), cameraMatrix, distCoeffs,
                     rvecs.at<cv::Vec3d>(i), tvecs.at<cv::Vec3d>(i));
    }

  private:
    cv::Mat &objPoints;
    cv::InputArrayOfArrays corners;
    cv::InputArray cameraMatrix, distCoeffs;
    cv::Mat &rvecs, tvecs;
  };

  // Variables for IPC
  void *memaddr = nullptr;
  cam_ipc_data_t *memdata = nullptr;
  shared_memory_object *segment = nullptr;
  named_semaphore *sem_filled = nullptr;
  named_semaphore *sem_empty = nullptr;
  mapped_region *memregion = nullptr;

  // ####################################################################################################
  // ####################################################################################################
  // ####################################################################################################

public:
  // ####################################################################################################
  //! Constructor
  FirstVision_Simplified(std::string const &instance) : jevois::StdModule(instance)
  {
    // Start IPC
    // Remove anything previously present
    shared_memory_object::remove("SharedMemoryCam");
    named_semaphore::remove("SemFilledCam");
    named_semaphore::remove("SemEmptyCam");

    // Create the shared memory object
    // Define the memory segment
    segment = new shared_memory_object(open_or_create, "SharedMemoryCam", read_write /*mode*/); // or could be

    // Open/create named semaphore
    sem_filled = new named_semaphore(open_or_create, "SemFilledCam", 0);
    sem_empty = new named_semaphore(open_or_create, "SemEmptyCam", 0);

    // Set size
    segment->truncate(sizeof(cam_ipc_data_t));

    // Map the shared memory
    memregion = new mapped_region(*segment /*what to map*/, read_write /*Map it as read-write*/);

    // Get the address of the mapped region
    memaddr = memregion->get_address();

    // Construct the shared structure in memory
    // I.e. assigning this pointer to the structure's address
    memdata = new (memaddr) cam_ipc_data_t;

    // Need to set sem_empty to high
    sem_empty->post();
  }

  // ####################################################################################################
  //! Virtual destructor for safe inheritance
  virtual ~FirstVision_Simplified()
  {

    // Remove the shared memory object
    shared_memory_object::remove("SharedMemoryCam");
    named_semaphore::remove("SemFilledCam");
    named_semaphore::remove("SemEmptyCam");
  }

  // ####################################################################################################
  //! Estimate 6D pose of detected objects, if dopose parameter is true, otherwise just 2D corners
  /*! Inspired from the ArUco module of opencv_contrib
        The corners array is always filled, but rvecs and tvecs only are if dopose is true */
  void estimatePose(std::vector<std::vector<cv::Point2f>> &corners, cv::OutputArray _rvecs,
                    cv::OutputArray _tvecs)
  {
    auto const osiz = objsize::get();

    // Get a vector of all our corners so we can map them to 3D and draw them:
    corners.clear();
    for (detection const &d : itsDetections)
    {
      corners.push_back(std::vector<cv::Point2f>());
      std::vector<cv::Point2f> &v = corners.back();
      for (auto const &p : d.hull)
        v.push_back(cv::Point2f(p));
    }

    // set coordinate system in the middle of the object, with Z pointing out
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-osiz.width * 0.5F, -osiz.height * 0.5F, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(-osiz.width * 0.5F, osiz.height * 0.5F, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(osiz.width * 0.5F, osiz.height * 0.5F, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(osiz.width * 0.5F, -osiz.height * 0.5F, 0);

    int nobj = (int)corners.size();
    _rvecs.create(nobj, 1, CV_64FC3);
    _tvecs.create(nobj, 1, CV_64FC3);
    cv::Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
    cv::parallel_for_(cv::Range(0, nobj), SinglePoseEstimationParallel(objPoints, corners, itsCamMatrix,
                                                                       itsDistCoeffs, rvecs, tvecs));
  }

  // ####################################################################################################
  //! Load camera calibration parameters
  void loadCameraCalibration(unsigned int w, unsigned int h)
  {
    camparams::freeze();

    std::string const cpf = std::string(JEVOIS_SHARE_PATH) + "/camera/" + camparams::get() +
                            std::to_string(w) + 'x' + std::to_string(h) + ".yaml";

    cv::FileStorage fs(cpf, cv::FileStorage::READ);
    if (fs.isOpened())
    {
      fs["camera_matrix"] >> itsCamMatrix;
      fs["distortion_coefficients"] >> itsDistCoeffs;
      LINFO("Loaded camera calibration from " << cpf);
    }
    else
      LFATAL("Failed to read camera parameters from file [" << cpf << "]");
  }

  // ####################################################################################################
  //! HSV object detector, we run several of those in parallel with different hsvcue settings
  void detect(cv::Mat const &imghsv, hsvcue const &hsv, jevois::RawImage *outimg = nullptr)
  {
    std::string str = "";

    // Apply gaussian blur
    cv::Mat imgblur;
    unsigned int len = blursize::get();
    len = len % 2 == 1 ? len : len + 1;
    // cv::GaussianBlur(imghsv, imgblur, cv::Size(len, len), blursigma::get());
    len = len >= 3 ? len : 3;
    cv::medianBlur(imghsv, imgblur, len);

    // Threshold the HSV image to only keep pixels within the desired HSV range:
    cv::Mat imgth;

    if (hsv.h_outer)
    {
      cv::Scalar rmin = hsv.rmin();
      cv::Scalar rmax = hsv.rmax();

      // [left_min, left_max] || [right_min, right_max]
      cv::Scalar left_min = rmin;
      left_min[0] = 0; // Need to set the min h to zero because of the overlap
      cv::Scalar left_max = rmax;
      left_max[0] = rmin[0];
      cv::Scalar right_min = rmin;
      right_min[0] = rmax[0];
      cv::Scalar right_max = rmax;
      right_max[0] = 179;

      // H: 0-179, S: 0-255, V:0-255
      cv::Mat imgth_lower;
      cv::Mat imgth_upper;
      cv::inRange(imgblur, left_min, left_max, imgth_lower);
      cv::inRange(imgblur, right_min, right_max, imgth_upper);
      cv::bitwise_or(imgth_lower, imgth_upper, imgth);
    }
    else
    {
      cv::Scalar const rmin = hsv.rmin(), rmax = hsv.rmax();
      cv::inRange(imgblur, rmin, rmax, imgth);
    }

    // Apply morphological operations to cleanup the image noise:
    if (itsErodeElement.empty() == false)
      cv::erode(imgth, imgth, itsErodeElement);
    if (itsDilateElement.empty() == false)
      cv::dilate(imgth, imgth, itsDilateElement);

    // Detect objects by finding contours:
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgth, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    str += jevois::sformat("N=%03d ", hierarchy.size());

    // Copy thresholded image to the output image
    if (outimg && outimg->valid())
    {
      if (outimg->width == 2 * imgth.cols)
      {
        jevois::rawimage::pasteGreyToYUYV(imgth, *outimg, imgth.cols, 0);
      }
    }

    int num_contours_with_child = 0;

    // Identify the "good" objects:
    std::string str_debug, str_flags, beststr_flags, beststr_debug;
    if (hierarchy.size() > 0 && hierarchy.size())
    {
      // Goes to the next index of the same hierarchy
      // Changed to only select the countours with at least one child contour
      for (int index = 0; index >= 0; index = hierarchy[index][0])
      {
        // Check that this contour has a child contour
        if (hierarchy[index][2] >= 0)
        {

          // Increment the counter
          num_contours_with_child += 1;

          // Keep track of our best detection so far:
          if (str_flags.length() > beststr_flags.length())
          {
            beststr_flags = str_flags;
            beststr_debug = str_debug;
          }
          str_flags.clear();
          str_debug.clear();

          // Let's examine the contour and child
          std::vector<cv::Point> const &c_parent = contours[index];
          std::vector<cv::Point> const &c_child = contours[hierarchy[index][2]];

          // Compute convex hull:
          std::vector<cv::Point> raw_hull_child, raw_hull_parent, hull_child, hull_parent;
          cv::convexHull(c_child, raw_hull_child, true);   // true --> clockwise
          cv::convexHull(c_parent, raw_hull_parent, true); // true --> clockwise
          double const childhullperi = cv::arcLength(raw_hull_child, true);
          cv::approxPolyDP(raw_hull_child, hull_child, epsilon::get() * childhullperi * 3.0, true);
          cv::approxPolyDP(raw_hull_parent, hull_parent, epsilon::get() * childhullperi * 3.0, true);

          // Is it the right shape?
          str_debug += "HP=" + std::to_string(hull_parent.size());
          str_debug += "HC=" + std::to_string(hull_child.size());
          str_flags += "-";
          if (hull_child.size() != 4 || (hull_parent.size() < 4 && hull_parent.size() > 6))
            continue;
          str_flags += "H,"; // Hull is quadrilateral

          // Look at the difference between the convex hull and the contour
          // Look at the difference between the two hulls or between the two contours

          // Compute contour area:
          double const area_parent = cv::contourArea(c_parent, false);
          double const area_child = cv::contourArea(c_child, false);
          double const area_diff = area_parent - area_child;

          int const fill_ratio = int(area_diff / area_parent * 100.0 + 0.4999);
          str_debug += ",F=" + std::to_string(fill_ratio); // fill ratio
          if (fill_ratio > fillratio_max::get() || fill_ratio < fillratio_min::get())
            continue;
          str_flags += "F,"; // Fill is ok

          // Not used
          std::vector<cv::Point> c_approx_child;
          double const childcontourperi = cv::arcLength(c_child, true);
          cv::approxPolyDP(c_child, c_approx_child, epsilon::get() * childcontourperi, true);

          // Compute contour shape error:
          const int shape_error = int(100.0 * cv::matchShapes(c_child, hull_child, cv::CONTOURS_MATCH_I1, 0.0));
          str_debug += ",SE=" + std::to_string(shape_error); // fill ratio
          if (shape_error > shapeerror_max::get())
            continue;
          str_flags += "E,"; // Shape error is ok

          // This detection is a keeper:
          str_flags += "OK";

          // TODO: Copy for now, but change to no copy later
          detection d;
          d.contour = c_child;
          d.approx = c_approx_child;
          d.hull = hull_child;
          d.serr = shape_error;
          d.fill_ratio = fill_ratio;
          std::lock_guard<std::mutex> _(itsDetMtx);
          itsDetections.push_back(d);

          // Draw the contour for each good detection
          if (outimg && outimg->valid() && outimg->width == 2 * imgth.cols)
          {

            // Pointer to the output image
            cv::Mat outuc2(outimg->height, outimg->width, CV_8UC2, outimg->pixelsw<unsigned char>());

            // Draw this contour
            std::vector<std::vector<cv::Point>> draw_contours; // An array of contours
            draw_contours.push_back(d.hull);
            draw_contours.push_back(d.contour);
            draw_contours.push_back(d.approx);
            cv::drawContours(outuc2, draw_contours, 0, jevois::yuyv::LightGreen, 2, 8, cv::noArray(), INT_MAX, cv::Point(imgth.cols, 0));
            cv::drawContours(outuc2, draw_contours, 1, jevois::yuyv::MedPurple, 1, 8, cv::noArray(), INT_MAX, cv::Point(imgth.cols, 0));
            cv::drawContours(outuc2, draw_contours, 2, jevois::yuyv::MedPurple, 1, 8, cv::noArray(), INT_MAX, cv::Point(imgth.cols, 0));

            jevois::rawimage::drawCircle(*outimg, d.hull[0].x + imgth.cols, d.hull[0].y, 10,
                                         2, jevois::yuyv::LightGreen);

            jevois::rawimage::drawCircle(*outimg, d.hull[1].x + imgth.cols, d.hull[1].y, 5,
                                         2, jevois::yuyv::LightGreen);
          }

          if (str_flags.length() > beststr_flags.length())
          {
            beststr_flags = str_flags;
            beststr_debug = str_debug;
          }
        }
      }
    }

    str += jevois::sformat("NC=%03d ", num_contours_with_child);

    // Display any results requested by the users:
    if (outimg && outimg->valid())
    {
      jevois::rawimage::writeText(*outimg, str + beststr_flags, 3, 20, jevois::yuyv::White);
      jevois::rawimage::writeText(*outimg, beststr_debug, 3, 20 + 12 * 2, jevois::yuyv::White);
    }
  }

  // ####################################################################################################
  //! Initialize (e.g., if user changes cue params)
  void updateHSV()
  {
    if (itsHSV.empty() || itsCueChanged)
    {
      // Initialize or reset because of user parameter change:
      itsHSV.clear();
      itsCueChanged = false;
      hsvcue cue(hmin::get(), hmax::get(), smin::get(), smax::get(), vmin::get(), vmax::get(), houter::get());
      itsHSV.push_back(cue);
    }
  }

  // ####################################################################################################
  //! Learn and update our HSV ranges
  // void learnHSV(size_t nthreads, cv::Mat const & imgbgr, jevois::RawImage *outimg = nullptr)
  // {
  //   int const w = imgbgr.cols, h = imgbgr.rows;

  //   // Compute the median filtered BGR image in a thread:
  //   cv::Mat medimgbgr;
  //   auto median_fut = std::async(std::launch::async, [&](){ cv::medianBlur(imgbgr, medimgbgr, 3); } );

  //   // Get all the cleaned-up contours:
  //   std::vector<std::vector<cv::Point> > contours;
  //   for (detection const & d : itsDetections) contours.push_back(d.contour);

  //   // If desired, draw all contours:
  //   std::future<void> drawc_fut;
  //   if (debug::get() && outimg && outimg->valid())
  //     drawc_fut = std::async(std::launch::async, [&]() {
  //         // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
  //         cv::Mat outuc2(outimg->height, outimg->width, CV_8UC2, outimg->pixelsw<unsigned char>());
  //         cv::drawContours(outuc2, contours, -1, jevois::yuyv::LightPink, 2);
  //       } );

  //   // Draw all the filled contours into a binary mask image:
  //   cv::Mat mask(h, w, CV_8UC1, (unsigned char)0);
  //   cv::drawContours(mask, contours, -1, 255, -1); // last -1 is for filled

  //   // Wait until median filter is done:
  //   median_fut.get();

  //   // Compute mean and std BGR values inside objects:
  //   cv::Mat mean, std;
  //   cv::meanStdDev(medimgbgr, mean, std, mask);

  //   // Convert to HSV:
  //   cv::Mat bgrmean(2, 1, CV_8UC3); bgrmean.at<cv::Vec3b>(0, 0) = mean; bgrmean.at<cv::Vec3b>(1, 0) = std;
  //   cv::Mat hsvmean; cv::cvtColor(bgrmean, hsvmean, cv::COLOR_BGR2HSV);

  //   cv::Vec3b hsv = hsvmean.at<cv::Vec3b>(0, 0);
  //   int H = hsv.val[0], S = hsv.val[1], V = hsv.val[2];

  //   cv::Vec3b sighsv = hsvmean.at<cv::Vec3b>(1, 0);
  //   int sH = sighsv.val[0], sS = sighsv.val[1], sV = sighsv.val[2];

  //   // Set the new measurements:
  //   itsKalH->set(H); itsKalS->set(S); itsKalV->set(V);

  //   if (nthreads > 2)
  //   {
  //     float const eta = 0.4F;
  //     itsHSV[2].sih = (1.0F - eta) * itsHSV[2].sih + eta * sH;
  //     itsHSV[2].sis = (1.0F - eta) * itsHSV[2].sis + eta * sS;
  //     itsHSV[2].siv = (1.0F - eta) * itsHSV[2].siv + eta * sV;
  //     itsHSV[2].fix();
  //   }

  //   // note: drawc_fut may block us here until it is complete.
  // }

  // ####################################################################################################
  //! Update the morphology structuring elements if needed
  void updateStructuringElements()
  {
    int e = erodesize::get();
    if (e != itsErodeElement.cols)
    {
      if (e)
        itsErodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(e, e));
      else
        itsErodeElement.release();
    }

    int d = dilatesize::get();
    if (d != itsDilateElement.cols)
    {
      if (d)
        itsDilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(d, d));
      else
        itsDilateElement.release();
    }
  }

  // ####################################################################################################
  //! Processing function, no USB video output
  virtual void process(jevois::InputFrame &&inframe) override
  {
    static jevois::Timer timer("processing");

    // Wait for next available camera image. Any resolution ok:
    jevois::RawImage inimg = inframe.get();
    unsigned int const w = inimg.width, h = inimg.height;

    timer.start();

    // Load camera calibration if needed:
    if (itsCamMatrix.empty())
      loadCameraCalibration(w, h);

    // Convert input image to BGR24, then to HSV:
    cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
    cv::Mat imghsv;
    cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

    // Make sure our HSV range parameters are up to date:
    updateHSV();

    // Clear any old detections and get ready to parallelize the detection work:
    itsDetections.clear();
    updateStructuringElements();

    // Launch our workers: run nthreads-1 new threads, and last worker in our current thread:
    // std::vector<std::future<void> > dfut;
    // for (size_t i = 0; i < nthreads - 1; ++i)
    //   dfut.push_back(std::async(std::launch::async, [&](size_t tn) { detect(imghsv, tn, 3, h+2); }, i));
    // detect(imghsv, nthreads - 1, 3, h+2);

    // // Wait for all threads to complete:
    // for (auto & f : dfut) try { f.get(); } catch (...) { jevois::warnAndIgnoreException(); }
    detect(imghsv, itsHSV[0]);

    // Let camera know we are done processing the input image:
    inframe.done();

    // Clean up the detections by eliminating duplicates:
    // cleanupDetections();

    // Learn the object's HSV value over time:
    // auto learn_fut = std::async(std::launch::async, [&]() { learnHSV(nthreads, imgbgr); });

    // Map to 6D (inverse perspective):
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    estimatePose(corners, rvecs, tvecs);

    // If there is at least one detection
    if (rvecs.size() && tvecs.size())
    {
      // Only sending the first detection
      // Convert the rodirigues rotation vector to a center-wise vector
      cv::Mat rotMat(3, 3, CV_64F);
      cv::Rodrigues(rvecs.at(0), rotMat);
      cv::Vec3d body_axis(0, 0, 1);
      cv::Mat3d cvec_m = rotMat * cv::Mat(body_axis); // Vector from the centre of the gate pointing outwards in cam frame
      cv::Vec3d cvec(cvec_m.at<cv::Vec3d>());
      sendDataIPC(cvec, tvecs.at(0));
    }

    // Send all serial messages:
    // sendAllSerial(w, h, corners, rvecs, tvecs);

    // Wait for all threads:
    // try { learn_fut.get(); } catch (...) { jevois::warnAndIgnoreException(); }

    // Show processing fps:
    timer.stop();
  }

  // ####################################################################################################
  //! Processing function, with USB video output
  virtual void process(jevois::InputFrame &&inframe, jevois::OutputFrame &&outframe) override
  {
    static jevois::Timer timer("processing");

    // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
    jevois::RawImage inimg = inframe.get();
    unsigned int const w = inimg.width, h = inimg.height;
    inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

    timer.start();

    // Load camera calibration if needed:
    if (itsCamMatrix.empty())
      loadCameraCalibration(w, h);

    // While we process it, start a thread to wait for output frame and paste the input image into it:
    jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
    auto paste_fut = std::async(std::launch::async, [&]() {
      outimg = outframe.get();
      outimg.require("output", outimg.width, h, inimg.fmt);
      if (outimg.width != w && outimg.width != w * 2)
        LFATAL("Output image width should be 1x or 2x input width");
      jevois::rawimage::paste(inimg, outimg, 0, 0);
      jevois::rawimage::drawFilledRect(outimg, 0, h, outimg.width, outimg.height - h, jevois::yuyv::Black);
    });

    // Convert input image to BGR24, then to HSV:
    cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
    cv::Mat imghsv;
    cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

    // Make sure our HSV range parameters are up to date:
    updateHSV();

    // Clear any old detections and get ready to parallelize the detection work:
    itsDetections.clear();
    updateStructuringElements();

    // Launch our workers: run nthreads-1 new threads, and last worker in our current thread:
    // std::vector<std::future<void> > dfut;
    // for (size_t i = 0; i < nthreads - 1; ++i)
    //   dfut.push_back(std::async(std::launch::async, [&](size_t tn) { detect(imghsv, tn, 3, h+2, &outimg); }, i));
    detect(imghsv, itsHSV[0], &outimg);

    // Wait for all threads to complete:
    // for (auto & f : dfut) try { f.get(); } catch (...) { jevois::warnAndIgnoreException(); }

    // Wait for paste to finish up:
    paste_fut.get();

    // Let camera know we are done processing the input image:
    inframe.done();

    // Clean up the detections by eliminating duplicates:
    // cleanupDetections();

    // Learn the object's HSV value over time:
    // auto learn_fut = std::async(std::launch::async, [&]() { learnHSV(nthreads, imgbgr, &outimg); });

    // Map to 6D (inverse perspective):
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    estimatePose(corners, rvecs, tvecs);

    // cv::Vec3d rvec_test(1, 2, 3);
    // cv::Vec3d tvec_test(4, 5, 6);

    // If there is at least one detection
    if (rvecs.size() && tvecs.size())
    {
      // Only sending the first detection
      // Convert the rodirigues rotation vector to a center-wise vector
      cv::Mat rotMat(3, 3, CV_64F);
      cv::Rodrigues(rvecs.at(0), rotMat);
      cv::Vec3d body_axis(0, 0, 1);
      cv::Mat3d cvec_m = rotMat * cv::Mat(body_axis); // Vector from the centre of the gate pointing outwards in cam frame
      cv::Vec3d cvec(cvec_m.at<cv::Vec3d>());
      sendDataIPC(cvec, tvecs.at(0));
    }

    // Draw all detections in 3D:
    drawDetections(outimg, corners, rvecs, tvecs);

    // Show number of detected objects:
    jevois::rawimage::writeText(outimg, "Detected " + std::to_string(itsDetections.size()) + " objects.",
                                w + 3, 3, jevois::yuyv::White);

    // Wait for all threads:
    // try { learn_fut.get(); } catch (...) { jevois::warnAndIgnoreException(); }

    // Show processing fps:
    std::string const &fpscpu = timer.stop();
    jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

    // Send the output image with our processing results to the host over USB:
    outframe.send();
  }

  bool sendDataIPC(const cv::Vec3d &rvec, const cv::Vec3d &tvec, const long int proc_time = 0)
  {

    // Make sure that the previous is not in the process of being read
    // Which would be that
    // sem_filled low and sem_empty low --> started reading, not done yet, ---> skip
    // sem_filled low and sem_empty high --> started and done reading/no data set --> normal
    // sem_filled high, sem_empty low --> not started reading --> reset then normal
    // sem_filled high, sem_empty high --> not possible
    if (sem_filled->try_wait())
    {
      if (sem_empty->try_wait())
      {
        // should not be possible to reach here
      }
      else
      {
        // not started reading
        // Replace the old data with the new data
      }
    }
    else
    {
      if (sem_empty->try_wait())
      {
        // started and done reading --> normal
      }
      else
      {
        // started reading, not done yet
        // Should be unlikely that this occurs, provided the time between
        // getting new data is slower than the time to read the old data
        // could put a delay in here, but for now a skip should suffice.
        // skip
        return false;
      }
    }

    // Copy the rotation and translation vectors into the send data memory
    memdata->translation[0] = tvec[0];
    memdata->translation[1] = tvec[1];
    memdata->translation[2] = tvec[2];

    memdata->rotation[0] = rvec[0];
    memdata->rotation[1] = rvec[1];
    memdata->rotation[2] = rvec[2];

    memdata->proc_time = proc_time;

    // Data is now filled
    sem_filled->post();

    return true;
  }

  // ####################################################################################################
  void drawDetections(jevois::RawImage &outimg, std::vector<std::vector<cv::Point2f>> corners,
                      std::vector<cv::Vec3d> const &rvecs, std::vector<cv::Vec3d> const &tvecs)
  {
    auto const osiz = objsize::get();
    float const w = osiz.width, h = osiz.height;
    int nobj = int(corners.size());

    // This code is like drawDetectedMarkers() in cv::aruco, but for YUYV output image:
    if (rvecs.empty())
    {
      // We are not doing 3D pose estimation. Just draw object outlines in 2D:
      for (int i = 0; i < nobj; ++i)
      {
        std::vector<cv::Point2f> const &obj = corners[i];

        // draw marker sides:
        for (int j = 0; j < 4; ++j)
        {
          cv::Point2f const &p0 = obj[j];
          cv::Point2f const &p1 = obj[(j + 1) % 4];
          jevois::rawimage::drawLine(outimg, int(p0.x + 0.5F), int(p0.y + 0.5F),
                                     int(p1.x + 0.5F), int(p1.y + 0.5F), 1, jevois::yuyv::LightPink);
          //jevois::rawimage::writeText(outimg, std::to_string(j),
          //			      int(p0.x + 0.5F), int(p0.y + 0.5F), jevois::yuyv::White);
        }
      }
    }
    else
    {
      // Show trihedron and parallelepiped centered on object:
      float const hw = w * 0.5F, hh = h * 0.5F, dd = -0.5F * std::max(w, h);

      for (int i = 0; i < nobj; ++i)
      {
        // Project axis points:
        std::vector<cv::Point3f> axisPoints;
        axisPoints.push_back(cv::Point3f(0.0F, 0.0F, 0.0F));
        axisPoints.push_back(cv::Point3f(hw, 0.0F, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, hh, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, 0.0F, dd));

        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsCamMatrix, itsDistCoeffs, imagePoints);

        // Draw axis lines:
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[1].x + 0.5F), int(imagePoints[1].y + 0.5F),
                                   2, jevois::yuyv::MedPurple);
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[2].x + 0.5F), int(imagePoints[2].y + 0.5F),
                                   2, jevois::yuyv::MedGreen);
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[3].x + 0.5F), int(imagePoints[3].y + 0.5F),
                                   2, jevois::yuyv::MedGrey);

        // Also draw a parallelepiped:
        std::vector<cv::Point3f> cubePoints;
        cubePoints.push_back(cv::Point3f(-hw, -hh, 0.0F));
        cubePoints.push_back(cv::Point3f(hw, -hh, 0.0F));
        cubePoints.push_back(cv::Point3f(hw, hh, 0.0F));
        cubePoints.push_back(cv::Point3f(-hw, hh, 0.0F));
        cubePoints.push_back(cv::Point3f(-hw, -hh, dd));
        cubePoints.push_back(cv::Point3f(hw, -hh, dd));
        cubePoints.push_back(cv::Point3f(hw, hh, dd));
        cubePoints.push_back(cv::Point3f(-hw, hh, dd));

        std::vector<cv::Point2f> cuf;
        cv::projectPoints(cubePoints, rvecs[i], tvecs[i], itsCamMatrix, itsDistCoeffs, cuf);

        // Round all the coordinates:
        std::vector<cv::Point> cu;
        for (auto const &p : cuf)
          cu.push_back(cv::Point(int(p.x + 0.5F), int(p.y + 0.5F)));

        // Draw parallelepiped lines:
        jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[1].x, cu[1].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[2].x, cu[2].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[3].x, cu[3].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[0].x, cu[0].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[4].x, cu[4].y, cu[5].x, cu[5].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[5].x, cu[5].y, cu[6].x, cu[6].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[6].x, cu[6].y, cu[7].x, cu[7].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[7].x, cu[7].y, cu[4].x, cu[4].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[4].x, cu[4].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[5].x, cu[5].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[6].x, cu[6].y, 1, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[7].x, cu[7].y, 1, jevois::yuyv::LightGreen);
      }
    }
  }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(FirstVision_Simplified);
