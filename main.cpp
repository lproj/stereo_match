/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Author: Luca Risolia <info@linux-projects.org>
*/

#include <boost/program_options.hpp>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {
struct program_options_t {
  fs::path leftimg, rightimg;
  int mindisp;
  unsigned numdisp, blocksize;
};

std::optional<program_options_t> parse_program_options(int argc, char **argv) {
  program_options_t opts;

  po::options_description desc{"Program options"};
  desc.add_options()("mindisp,m",
                     po::value<int>(&opts.mindisp)->default_value(0),
                     "minimum disparity")(
      "numdisp,n", po::value<unsigned>(&opts.numdisp)->default_value(64),
      "number of disparities")(
      "blocksize,b", po::value<unsigned>(&opts.blocksize)->default_value(21),
      "block size (must be an odd number)")(
      "left,l",
      po::value<std::string>()->required()->notifier(
          [&opts](const auto &v) { opts.leftimg = v; }),
      "path to the left image")(
      "right,r",
      po::value<std::string>()->required()->notifier(
          [&opts](const auto &v) { opts.rightimg = v; }),
      "path to the right image");

  po::options_description other_desc{"Other options"};
  other_desc.add_options()("help,h", "print help screen and exit");
  desc.add(other_desc);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << '\n';
    return {};
  }

  po::notify(vm);

  return opts;
}

using stereo_t = std::pair<cv::Mat, cv::Mat>;

} // namespace

stereo_t load_imgs(const fs::path &left, const fs::path &right) {
  return {cv::imread(left.string(), cv::IMREAD_GRAYSCALE),
          cv::imread(right.string(), cv::IMREAD_GRAYSCALE)};
}

cv::Mat compute_dispmap(const stereo_t &stereo_imgs, unsigned int numdisp,
                        int mindisp = 0, int blocksize = 21) {
  const auto &[limg, rimg] = stereo_imgs;
  cv::Mat dispmap(limg.rows, limg.cols, CV_8UC1);
  for (unsigned w = blocksize / 2, y = w; y < limg.rows - w - 1; y++) {
    for (int maxdisp = numdisp + mindisp, x = maxdisp + w;
         x < limg.cols + mindisp - w - 1; x++) {
      cv::Rect block{cv::Point(x - w, y - w), cv::Size(blocksize, blocksize)};
      // std::cout << block << " in ";
      const auto feature = limg(block);
      cv::Rect slice{cv::Point(x - w - maxdisp, y - w),
                     cv::Point(x + w + 1 - mindisp, y + w + 1)};
      // std::cout << slice << '\n';
      const auto img = rimg(slice);
      cv::Mat nccs;
      cv::matchTemplate(img, feature, nccs, cv::TM_CCORR_NORMED);
      cv::Point max{};
      cv::minMaxLoc(nccs, nullptr, nullptr, nullptr, &max, cv::noArray());
      dispmap.at<uchar>(y, x) = max.x;
    }
  }
  return dispmap;
}

void show_dispmap(cv::Mat dispmap) {
  cv::Mat ndispmap;
  cv::normalize(dispmap, ndispmap, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::applyColorMap(ndispmap, dispmap, cv::COLORMAP_JET);
  cv::imshow("disparity_map", dispmap);
  cv::waitKey();
  cv::destroyWindow("disparity_map");
}

int main(int argc, char **argv) try {
  const auto opts = parse_program_options(argc, argv);
  if (!opts) {
    return EXIT_SUCCESS;
  }

  const auto stereoimgs = load_imgs(opts->leftimg, opts->rightimg);
  const auto dispmap = compute_dispmap(stereoimgs, opts->numdisp, opts->mindisp,
                                       opts->blocksize);
  show_dispmap(dispmap);

  return EXIT_SUCCESS;

} catch (const po::error &err) {
  std::cerr << err.what() << '\n';
  return EXIT_FAILURE;

} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return EXIT_FAILURE;

} catch (...) {
  std::cerr << "unknown exception\n";
  return EXIT_FAILURE;
}
