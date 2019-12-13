// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <condition_variable>
#include <mutex>
#include <thread>

#include "dataset/track_dataset.hpp"
#include "tracker/tracking_manager.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/plot.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"

#include "util/logger.hpp"

#ifdef __clang__
namespace fs
{
class path
{
public:
  explicit path(const std::string & p)
  : path_(p) {}
  bool is_absolute() {return path_[0] == '/';}

private:
  std::string path_;
};
}  // namespace fs
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class Streamer_node
{
public:
  Streamer_node() {}

  void initialDataset(
    std::string path, datasets::dsType type,
    std::string dsName)
  {
    ds_ = ds_->create(type);
    ds_->load(path);
    ds_->initDataset(dsName);
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
  }

public:
  void emitFrame();
  void emitDetect();

  int getFrameIdx(){return ds_->getFrameIdx();}

  std::thread emitThread;
  std::mutex g_runlock;
  std::condition_variable g_frame_signal;

protected:
  const std::string window = "Tracking API";
  const cv::Scalar gtColor = cv::Scalar(0, 255, 0);
  cv::Ptr<datasets::trDataset> ds_;
  cv::Mat frame_, image_;

  int num_present_ = 0;
  int num_response_ = 0;
  int num_corr_ = 0;
  int num_corr_thd_ = 0;

  tracker::TrackingManager tm_;
};

void Streamer_node::emitFrame()
{
  std::unique_lock<std::mutex> locker(g_runlock);

  if (ds_->getNextFrame(frame_)) {
    int frameId = ds_->getFrameIdx();
    TRACE_INFO("regression rgb frameId:%d", frameId);
    timespec stamp;
    stamp.tv_sec = frameId / 1000 + 1;
    stamp.tv_nsec = (frameId % 1000) * 1e6;

    cv::Mat frame_emit = frame_.clone();
    std::shared_ptr<sFrame> frame = std::make_shared<sFrame>(frame_emit, stamp);
    tm_.track(frame);

    putText(frame_, std::to_string(frameId), cv::Point(0, 15),
      cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    for (auto it : ds_->getIdxGT(frameId)) {
      cv::Rect2d gt_roi = it.bb;
      rectangle(frame_, gt_roi, cv::Scalar(0, 255, 0), 1, cv::LINE_8);
    }

    const std::vector<std::shared_ptr<tracker::Tracking>> trackings =
      tm_.getTrackedObjs();
    for (auto t : trackings) {
      cv::Rect2d r = t->getTrackedRect();
      int id = t->getTrackingId();

      cv::Point point = r.tl();
      putText(frame_, std::to_string(id), point, cv::FONT_HERSHEY_PLAIN, 1,
        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
      rectangle(frame_, r, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
      TRACE_INFO("track id(%d), rect(%d)", id, r);

      //cv::Mat covar = 64*t->tracker_->getCovar().clone();
      cv::Mat covar = t->predictCovar_;
      cv::Mat eigVal, eigVec;
      cv::eigen(covar, eigVal, eigVec);
      cv::sqrt(eigVal, eigVal);
//    cv::ellipse( img, (r.tl() + r.br())/2, Size(20,35), 0, 0, 360, white, -1, 8, 0 );

      // 9.210340f from Chi-square distribution
      //eigVal = eigVal * sqrt(9.210340f);
      // 2.8f from Chi-square distribution for 80% possibility
      eigVal = eigVal * sqrt(2.8f);
      cv::Point center = (r.tl() + r.br())/2;
      cv::Point pa_major1, pa_major2;
      pa_major1.x = center.x + eigVal.at<float>(0) * eigVec.row(0).at<float>(0);
      pa_major1.y = center.y + eigVal.at<float>(0) * eigVec.row(0).at<float>(1);
      pa_major2.x = center.x - eigVal.at<float>(0) * eigVec.row(0).at<float>(0);
      pa_major2.y = center.y - eigVal.at<float>(0) * eigVec.row(0).at<float>(1);
      cv::Point pb_major1, pb_major2;
      pb_major1.x = center.x + eigVal.at<float>(1) * eigVec.row(1).at<float>(0);
      pb_major1.y = center.y + eigVal.at<float>(1) * eigVec.row(1).at<float>(1);
      pb_major2.x = center.x - eigVal.at<float>(1) * eigVec.row(1).at<float>(0);
      pb_major2.y = center.y - eigVal.at<float>(1) * eigVec.row(1).at<float>(1);
      cv::line(frame_, pa_major1, pa_major2, cv::Scalar(255), 1, cv::LINE_8);
      cv::line(frame_, pb_major1, pb_major2, cv::Scalar(255), 1, cv::LINE_8);
    }

    for (auto t : trackings) {
      std::vector<tracker::Traj> trajs = t->getTrajs();
      cv::Point2d centerpre, center;
      if (trajs.size() <= 0)
         continue;

      centerpre = (trajs[0].rect_.tl() + trajs[0].rect_.br())/2;

      for (auto &traj : trajs)
      {
       //rectangle(image, traj.rect_, Scalar(0, 255, 0), 2, 1);
        center = (traj.rect_.tl() + traj.rect_.br())/2;
        cv::circle(frame_, center, 4, cv::Scalar(0, 0, 255));
        cv::line(frame_, centerpre,center, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        centerpre = center;
      }
    }

#ifndef NDEBUG
    imshow(window, frame_);
#endif
    num_present_++;

  } else {
    TRACE_INFO("\n-----------------Test ended!-----------------\n");
  }
}

void Streamer_node::emitDetect()
{
  int frameId = ds_->getFrameIdx();
  if ((((frameId % 3) == 0) && (frameId > 2)) || (frameId == 0)) {
    if (frameId == 0) {frameId = 1;}

    cv::Mat frame_det;

    if (frameId > 1) {frameId -= 1;}
    timespec stamp;
    stamp.tv_sec = frameId / 1000 + 1;
    stamp.tv_nsec = (frameId % 1000) * 1e6;

    ds_->getIdxFrame(frame_det, frameId);
    cv::Mat frame_draw = frame_det.clone();

    putText(frame_draw, std::to_string(frameId), cv::Point(0, 15),
      cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    std::vector<Object> objs_in_boxes;
    for (auto t : ds_->getIdxGT(frameId)) {
      cv::Rect2d roi = t.bb;
      Object c_obj;
      c_obj.Category_ = "test_traj";
      c_obj.Confidence_ = 100;
      c_obj.Stamp_ = stamp;
      c_obj.BoundBox_.x = roi.x;
      c_obj.BoundBox_.y = roi.y;
      c_obj.BoundBox_.width = roi.width;
      c_obj.BoundBox_.height = roi.height;

      rectangle(frame_draw, roi, cv::Scalar(0, 255, 0), 1, cv::LINE_8);

      objs_in_boxes.push_back(c_obj);
    }

    const std::vector<std::shared_ptr<tracker::Tracking>> trackings =
      tm_.getTrackedObjs();
    for (auto t : trackings) {
      int id = t->getTrackingId();

      tracker::Traj traj;
      bool ret = t->getTraj(stamp, traj);
      if (!ret) {continue;}

      cv::Point point = traj.rect_.tl();
      putText(frame_draw, std::to_string(id), point, cv::FONT_HERSHEY_PLAIN, 1,
        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
      rectangle(frame_draw, traj.rect_, cv::Scalar(255, 0, 0), 1, cv::LINE_8);

      cv::Mat covar = traj.covar_;
      cv::Mat eigVal, eigVec;
      cv::eigen(covar, eigVal, eigVec);
      cv::sqrt(eigVal, eigVal);
//    cv::ellipse( img, (r.tl() + r.br())/2, Size(20,35), 0, 0, 360, white, -1, 8, 0 );

      // 9.210340f from Chi-square distribution
      //eigVal = eigVal * sqrt(9.210340f);
      // 2.8f from Chi-square distribution for 80% possibility
      eigVal = eigVal * sqrt(2.8f);
      cv::Point center = (traj.rect_.tl() + traj.rect_.br())/2;
      cv::Point pa_major1, pa_major2;
      pa_major1.x = center.x + eigVal.at<float>(0) * eigVec.row(0).at<float>(0);
      pa_major1.y = center.y + eigVal.at<float>(0) * eigVec.row(0).at<float>(1);
      pa_major2.x = center.x - eigVal.at<float>(0) * eigVec.row(0).at<float>(0);
      pa_major2.y = center.y - eigVal.at<float>(0) * eigVec.row(0).at<float>(1);
      cv::Point pb_major1, pb_major2;
      pb_major1.x = center.x + eigVal.at<float>(1) * eigVec.row(1).at<float>(0);
      pb_major1.y = center.y + eigVal.at<float>(1) * eigVec.row(1).at<float>(1);
      pb_major2.x = center.x - eigVal.at<float>(1) * eigVec.row(1).at<float>(0);
      pb_major2.y = center.y - eigVal.at<float>(1) * eigVec.row(1).at<float>(1);
      cv::line(frame_draw, pa_major1, pa_major2, cv::Scalar(255), 1, cv::LINE_8);
      cv::line(frame_draw, pb_major1, pb_major2, cv::Scalar(255), 1, cv::LINE_8);


    }

    TRACE_INFO("regression detection frameId:%d", frameId);
    std::shared_ptr<sFrame> sframe_det =
      std::make_shared<sFrame>(frame_det, stamp);
    tm_.detectRecvProcess(sframe_det, objs_in_boxes);
#ifndef NDEBUG
    imshow("detection update", frame_draw);
#endif
  }
}

const std::string keys = {
  "{t tracker_algorithm |KCF     | Tracker algorithm }"
  "{p ds_path      || Dataset path }"
  "{n ds_name      || Dataset Name }"
  "{ds ds_type     |st_image| Dataset Type: st_image or mt_image}"
  "{g debug_file   |<none>  | logger config file }"
  "{f stop_frame   |0| stop by frame index}"
  "{h help     || show usage}"
};

int main(int argc, char * argv[])
{
  std::string dsPath, dsName, dType;
  datasets::dsType dsTpy;
  std::string algo;
  Streamer_node t_node;

  cv::CommandLineParser parser(argc, argv, keys);
  dsPath = parser.get<std::string>("ds_path");
  dsName = parser.get<std::string>("ds_name");

  if (parser.get<std::string>("ds_type") == "mt_image")
    dsTpy = datasets::dsMTImage;
  else
    dsTpy = datasets::dsSTImage;

  if (dsPath == "" || dsName == "") {
    TRACE_INFO("Please specfic below options:\n");
    parser.printMessage();
    return 0;
  }

  std::string dbgFile = parser.get<std::string>("debug_file");
  if (!dbgFile.empty())
    CONFIG_LOGGER_MODULES(dbgFile);

  int frame_stop = parser.get<int>("stop_frame");

  t_node.initialDataset(dsPath, dsTpy, dsName);

  t_node.emitDetect();
  t_node.emitFrame();

  t_node.g_frame_signal.notify_all();

  do {
    t_node.emitFrame();
    t_node.emitDetect();
  } while (t_node.getFrameIdx() < frame_stop);

  while (true) {
#ifndef NDEBUG
    int key = cv::waitKey(0);
    if (key == ' ') {
#endif
      t_node.emitFrame();
      t_node.emitDetect();
#ifndef NDEBUG
    } else if (key == 'q') {
      break;
    }
#endif
  }

  return 0;
}
