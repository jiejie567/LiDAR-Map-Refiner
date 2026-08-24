// Sparse execution adapter for the exact BALM2 eigenvalue derivatives.
//
// The equations in evaluate_exact_sparse are a sparsity-only transcription of
// hku-mars/BALM src/benchmark/bavoxel.hpp::VOX_HESS::acc_evaluate2 at commit
// 5dc1bf927fcb65ef17f0e687f553c234d6b17365. The optional dense audit invokes
// that unmodified official function and rejects any algebraic disagreement.

#include <chrono>
namespace ros {
class Time {
 public:
  static Time now() { return Time(); }
  double toSec() const {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
};
}  // namespace ros

#include "bavoxel.hpp"

#include <pcl/io/pcd_io.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

struct Observation {
  int pose_id = -1;
  PointCluster cluster;
};

struct PlaneFactor {
  std::vector<Observation> observations;
  double coefficient = 0.0;
};

struct Dataset {
  std::vector<double> timestamps;
  std::vector<IMUST> poses;
  std::vector<PlaneFactor> planes;
};

template <typename T>
void read_exact(std::istream& stream, T* value, std::size_t count = 1) {
  stream.read(reinterpret_cast<char*>(value), sizeof(T) * count);
  if (!stream) throw std::runtime_error("truncated association file");
}

Dataset load_dataset(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  char magic[8];
  read_exact(stream, magic, 8);
  if (std::string(magic, 8) != "GBALM2A1") {
    throw std::runtime_error("association magic mismatch");
  }
  std::uint32_t version, pose_count, plane_count;
  read_exact(stream, &version);
  read_exact(stream, &pose_count);
  read_exact(stream, &plane_count);
  if (version != 1 || pose_count < 2) throw std::runtime_error("unsupported data");
  Dataset data;
  data.timestamps.resize(pose_count);
  data.poses.resize(pose_count);
  for (std::uint32_t i = 0; i < pose_count; ++i) {
    read_exact(stream, &data.timestamps[i]);
    double rotation[9], translation[3];
    read_exact(stream, rotation, 9);
    read_exact(stream, translation, 3);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        data.poses[i].R(row, col) = rotation[row * 3 + col];
      }
      data.poses[i].p(row) = translation[row];
    }
  }
  data.planes.resize(plane_count);
  for (std::uint32_t plane = 0; plane < plane_count; ++plane) {
    std::uint32_t observation_count;
    read_exact(stream, &observation_count);
    auto& factor = data.planes[plane];
    factor.observations.resize(observation_count);
    for (auto& observation : factor.observations) {
      std::uint32_t pose_id, point_count;
      read_exact(stream, &pose_id);
      read_exact(stream, &point_count);
      if (pose_id >= pose_count || point_count == 0) {
        throw std::runtime_error("invalid observation");
      }
      observation.pose_id = static_cast<int>(pose_id);
      observation.cluster.N = static_cast<int>(point_count);
      read_exact(stream, observation.cluster.v.data(), 3);
      double second[9];
      read_exact(stream, second, 9);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          observation.cluster.P(row, col) = second[row * 3 + col];
        }
      }
      factor.coefficient += point_count;
    }
  }
  return data;
}

Dataset load_official_associations(
    const std::string& tum_path, const std::string& keyframe_dir,
    double root_voxel, double downsample_leaf, int octree_layers,
    double eigen_ratio, double terminal_eigen_ratio,
    int layer_point_threshold, int minimum_points) {
  std::ifstream trajectory_stream(tum_path);
  if (!trajectory_stream) throw std::runtime_error("cannot open " + tum_path);
  Dataset data;
  std::string line;
  while (std::getline(trajectory_stream, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    double timestamp, x, y, z, qx, qy, qz, qw;
    if (!(row >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      throw std::runtime_error("malformed TUM trajectory");
    }
    Eigen::Quaterniond quaternion(qw, qx, qy, qz);
    quaternion.normalize();
    IMUST pose;
    pose.R = quaternion.toRotationMatrix();
    pose.p = Eigen::Vector3d(x, y, z);
    data.timestamps.push_back(timestamp);
    data.poses.push_back(pose);
  }
  if (data.poses.size() < 2) throw std::runtime_error("trajectory too short");
  win_size = static_cast<int>(data.poses.size());
  voxel_size = root_voxel;
  layer_limit = std::clamp(octree_layers, 0, 3);
  min_ps = minimum_points;
  for (int layer = 0; layer < 4; ++layer) {
    eigen_value_array[layer] = static_cast<float>(
        layer == layer_limit ? terminal_eigen_ratio : eigen_ratio);
    layer_size[layer] = layer_point_threshold;
  }
  std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surface_map;
  for (int pose_id = 0; pose_id < win_size; ++pose_id) {
    const std::string path = keyframe_dir + "/" + std::to_string(pose_id) + ".pcd";
    pcl::PointCloud<PointType> cloud;
    if (pcl::io::loadPCDFile(path, cloud) != 0) {
      throw std::runtime_error("cannot load " + path);
    }
    // benchmark_realworld.cpp sends the raw per-frame clouds to cut_voxel().
    // Its 5 cm downsampling is visualization-only.  A non-positive leaf keeps
    // that official optimization path exactly; positive values are explicit
    // preprocessing ablations for a fair compute-budget comparison.
    if (downsample_leaf > 0.0) down_sampling_voxel(cloud, downsample_leaf);
    cut_voxel(surface_map, cloud, data.poses[pose_id], pose_id);
    if ((pose_id + 1) % 100 == 0 || pose_id + 1 == win_size) {
      std::cerr << "official_association loaded=" << (pose_id + 1)
                << "/" << win_size << " roots=" << surface_map.size() << "\n";
    }
  }
  VOX_HESS official_factors;
  for (auto& entry : surface_map) {
    entry.second->recut(win_size);
    entry.second->tras_opt(official_factors, win_size);
  }
  data.planes.reserve(official_factors.plvec_voxels.size());
  for (std::size_t plane_id = 0;
       plane_id < official_factors.plvec_voxels.size(); ++plane_id) {
    PlaneFactor factor;
    factor.coefficient = official_factors.coeffs[plane_id];
    const auto& clusters = *official_factors.plvec_voxels[plane_id];
    for (int pose_id = 0; pose_id < win_size; ++pose_id) {
      if (clusters[pose_id].N == 0) continue;
      factor.observations.push_back({pose_id, clusters[pose_id]});
    }
    data.planes.push_back(std::move(factor));
  }
  for (auto& entry : surface_map) delete entry.second;
  std::cerr << "official_association planes=" << data.planes.size() << "\n";
  return data;
}

std::uint64_t block_key(int row, int col) {
  if (row > col) std::swap(row, col);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32)
       | static_cast<std::uint32_t>(col);
}

using BlockMap = std::map<
    std::uint64_t, Matrix6d, std::less<std::uint64_t>,
    Eigen::aligned_allocator<std::pair<const std::uint64_t, Matrix6d>>>;

void add_block(BlockMap& blocks, int pose_a, int pose_b, const Matrix6d& value) {
  if (pose_a == 0 || pose_b == 0) return;
  int row = pose_a - 1;
  int col = pose_b - 1;
  Matrix6d oriented = value;
  if (row > col) {
    std::swap(row, col);
    oriented.transposeInPlace();
  }
  auto [it, inserted] = blocks.emplace(block_key(row, col), Matrix6d::Zero());
  it->second += oriented;
}

double evaluate_residual(
    const std::vector<IMUST>& poses, const std::vector<PlaneFactor>& planes) {
  double residual = 0.0;
  for (const auto& plane : planes) {
    PointCluster combined;
    for (const auto& observation : plane.observations) {
      PointCluster transformed;
      transformed.transform(observation.cluster, poses[observation.pose_id]);
      combined += transformed;
    }
    const Eigen::Vector3d center = combined.v / combined.N;
    const Eigen::Matrix3d covariance =
        combined.P / combined.N - center * center.transpose();
    residual += plane.coefficient
              * Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(covariance)
                    .eigenvalues()[0];
  }
  return residual;
}

double evaluate_exact_sparse(
    const std::vector<IMUST>& poses, const std::vector<PlaneFactor>& planes,
    Eigen::SparseMatrix<double>& hessian, Eigen::VectorXd& gradient) {
  const int variable_count = 6 * (static_cast<int>(poses.size()) - 1);
  gradient = Eigen::VectorXd::Zero(variable_count);
  BlockMap blocks;
  double residual = 0.0;
  for (const auto& plane : planes) {
    const int active_count = static_cast<int>(plane.observations.size());
    std::vector<PointCluster> transformed(active_count);
    PointCluster combined;
    for (int a = 0; a < active_count; ++a) {
      const auto& observation = plane.observations[a];
      transformed[a].transform(observation.cluster, poses[observation.pose_id]);
      combined += transformed[a];
    }
    const Eigen::Vector3d center = combined.v / combined.N;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(
        combined.P / combined.N - center * center.transpose());
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    const Eigen::Matrix3d eigenvectors = solver.eigenvectors();
    const Eigen::Vector3d normal = eigenvectors.col(0);
    const Eigen::Matrix3d normal_outer = normal * normal.transpose();
    Eigen::Matrix3d eigenvector_response = Eigen::Matrix3d::Zero();
    for (int axis = 1; axis < 3; ++axis) {
      const double gap = eigenvalues[0] - eigenvalues[axis];
      if (std::abs(gap) < 1e-12) continue;
      eigenvector_response += 2.0 / gap
          * eigenvectors.col(axis) * eigenvectors.col(axis).transpose();
    }
    const double total_count = combined.N;
    std::vector<Eigen::Matrix<double, 3, 6>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> A(active_count);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        viRiTuk(active_count);
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
        viRiTukukT(active_count);

    for (int a = 0; a < active_count; ++a) {
      const auto& observation = plane.observations[a];
      const PointCluster& cluster = observation.cluster;
      const IMUST& pose = poses[observation.pose_id];
      Eigen::Matrix3d vihat = hat(cluster.v);
      Eigen::Vector3d RiTuk = pose.R.transpose() * normal;
      Eigen::Matrix3d RiTukhat = hat(RiTuk);
      Eigen::Vector3d PiRiTuk = cluster.P * RiTuk;
      viRiTuk[a] = vihat * RiTuk;
      viRiTukukT[a] = viRiTuk[a] * normal.transpose();
      Eigen::Vector3d translation_center = pose.p - center;
      const double projected_translation = normal.dot(translation_center);
      Eigen::Matrix3d combo1 =
          hat(PiRiTuk) + vihat * projected_translation;
      Eigen::Vector3d combo2 =
          pose.R * cluster.v + cluster.N * translation_center;
      A[a].block<3, 3>(0, 0) =
          (pose.R * cluster.P + translation_center * cluster.v.transpose())
              * RiTukhat - pose.R * combo1;
      A[a].block<3, 3>(0, 3) =
          combo2 * normal.transpose() + combo2.dot(normal) * I33;
      A[a] /= total_count;
      const Vector6d jacobian = A[a].transpose() * normal;
      if (observation.pose_id > 0) {
        gradient.segment<6>(6 * (observation.pose_id - 1)) +=
            plane.coefficient * jacobian;
      }
      const Eigen::Matrix3d HRt = 2.0 / total_count
          * (1.0 - cluster.N / total_count) * viRiTukukT[a];
      Matrix6d block = A[a].transpose() * eigenvector_response * A[a];
      block.block<3, 3>(0, 0) +=
          2.0 / total_count * (combo1 - RiTukhat * cluster.P) * RiTukhat
          - 2.0 / (total_count * total_count)
              * viRiTuk[a] * viRiTuk[a].transpose()
          - 0.5 * hat(jacobian.head<3>());
      block.block<3, 3>(0, 3) += HRt;
      block.block<3, 3>(3, 0) += HRt.transpose();
      block.block<3, 3>(3, 3) += 2.0 / total_count
          * (cluster.N - cluster.N * cluster.N / total_count) * normal_outer;
      add_block(blocks, observation.pose_id, observation.pose_id,
                plane.coefficient * block);
    }
    for (int a = 0; a < active_count - 1; ++a) {
      const auto& first = plane.observations[a];
      for (int b = a + 1; b < active_count; ++b) {
        const auto& second = plane.observations[b];
        Matrix6d block = A[a].transpose() * eigenvector_response * A[b];
        block.block<3, 3>(0, 0) +=
            -2.0 / (total_count * total_count)
                * viRiTuk[a] * viRiTuk[b].transpose();
        block.block<3, 3>(0, 3) +=
            -2.0 * second.cluster.N / (total_count * total_count)
                * viRiTukukT[a];
        block.block<3, 3>(3, 0) +=
            -2.0 * first.cluster.N / (total_count * total_count)
                * viRiTukukT[b].transpose();
        block.block<3, 3>(3, 3) +=
            -2.0 * first.cluster.N * second.cluster.N
                / (total_count * total_count) * normal_outer;
        add_block(blocks, first.pose_id, second.pose_id,
                  plane.coefficient * block);
      }
    }
    residual += plane.coefficient * eigenvalues[0];
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(blocks.size() * 72);
  for (const auto& [key, block] : blocks) {
    const int block_row = static_cast<int>(key >> 32);
    const int block_col = static_cast<int>(key & 0xffffffffu);
    for (int row = 0; row < 6; ++row) {
      for (int col = 0; col < 6; ++col) {
        const double value = block(row, col);
        if (value == 0.0) continue;
        triplets.emplace_back(6 * block_row + row, 6 * block_col + col, value);
        if (block_row != block_col) {
          triplets.emplace_back(
              6 * block_col + col, 6 * block_row + row, value);
        }
      }
    }
  }
  hessian.resize(variable_count, variable_count);
  hessian.setFromTriplets(triplets.begin(), triplets.end());
  return residual;
}

void verify_unmodified_dense(const Dataset& data, double tolerance) {
  if (data.poses.size() > 100) {
    throw std::runtime_error("dense official audit is limited to 100 poses");
  }
  win_size = static_cast<int>(data.poses.size());
  std::vector<std::vector<PointCluster>> clusters(data.planes.size());
  std::vector<PointCluster> fixed(data.planes.size());
  VOX_HESS official;
  for (std::size_t plane = 0; plane < data.planes.size(); ++plane) {
    clusters[plane].resize(win_size);
    for (const auto& observation : data.planes[plane].observations) {
      clusters[plane][observation.pose_id] = observation.cluster;
    }
    official.plvec_voxels.push_back(&clusters[plane]);
    official.sig_vecs.push_back(&fixed[plane]);
    official.coeffs.push_back(data.planes[plane].coefficient);
  }
  Eigen::MatrixXd official_hessian(6 * win_size, 6 * win_size);
  Eigen::VectorXd official_gradient(6 * win_size);
  double official_residual = 0.0;
  official.acc_evaluate2(
      data.poses, 0, static_cast<int>(data.planes.size()),
      official_hessian, official_gradient, official_residual);

  // A full sparse transcription is obtained by prepending a dummy fixed pose;
  // its reduced system then contains every real pose variable.
  Dataset shifted = data;
  shifted.poses.insert(shifted.poses.begin(), IMUST());
  for (auto& plane : shifted.planes) {
    for (auto& observation : plane.observations) ++observation.pose_id;
  }
  Eigen::SparseMatrix<double> sparse_hessian;
  Eigen::VectorXd sparse_gradient;
  const double sparse_residual = evaluate_exact_sparse(
      shifted.poses, shifted.planes, sparse_hessian, sparse_gradient);
  Eigen::MatrixXd dense_sparse(sparse_hessian);
  const double hessian_error =
      (official_hessian - dense_sparse).cwiseAbs().maxCoeff();
  const double gradient_error =
      (official_gradient - sparse_gradient).cwiseAbs().maxCoeff();
  const double residual_error = std::abs(official_residual - sparse_residual);
  std::cerr << "dense_audit residual_error=" << residual_error
            << " gradient_error=" << gradient_error
            << " hessian_error=" << hessian_error << "\n";
  if (std::max({residual_error, gradient_error, hessian_error}) > tolerance) {
    throw std::runtime_error("sparse algebra differs from official BALM2 kernel");
  }
}

Eigen::Matrix3d exponential(const Eigen::Vector3d& omega) {
  return Exp(omega);
}

struct OptimizationReport {
  int attempted = 0;
  int accepted = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double final_damping = 0.0;
  double max_translation_update = 0.0;
  double max_rotation_update = 0.0;
};

OptimizationReport optimize(
    std::vector<IMUST>& poses, const std::vector<PlaneFactor>& planes,
    int max_iterations, double initial_damping) {
  OptimizationReport report;
  double damping = initial_damping;
  double nu = 2.0;
  bool recompute = true;
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  double current = evaluate_residual(poses, planes);
  report.initial_residual = current;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    ++report.attempted;
    if (recompute) {
      current = evaluate_exact_sparse(poses, planes, hessian, gradient);
    }
    Eigen::VectorXd diagonal = hessian.diagonal();
    std::vector<Eigen::Triplet<double>> damping_triplets;
    damping_triplets.reserve(diagonal.size());
    for (int i = 0; i < diagonal.size(); ++i) {
      damping_triplets.emplace_back(i, i, damping * diagonal[i]);
    }
    Eigen::SparseMatrix<double> damped(hessian.rows(), hessian.cols());
    damped.setFromTriplets(damping_triplets.begin(), damping_triplets.end());
    damped += hessian;
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(damped);
    solver.factorize(damped);
    if (solver.info() != Eigen::Success) {
      damping *= nu;
      nu *= 2.0;
      recompute = false;
      continue;
    }
    Eigen::VectorXd delta = solver.solve(-gradient);
    if (solver.info() != Eigen::Success || !delta.allFinite()) {
      damping *= nu;
      nu *= 2.0;
      recompute = false;
      continue;
    }
    std::vector<IMUST> candidate = poses;
    double max_rotation = 0.0;
    double max_translation = 0.0;
    for (int pose = 1; pose < static_cast<int>(poses.size()); ++pose) {
      const Eigen::Vector3d rotation = delta.segment<3>(6 * (pose - 1));
      const Eigen::Vector3d translation = delta.segment<3>(6 * (pose - 1) + 3);
      candidate[pose].R = poses[pose].R * exponential(rotation);
      candidate[pose].p = poses[pose].p + translation;
      max_rotation = std::max(max_rotation, rotation.norm());
      max_translation = std::max(max_translation, translation.norm());
    }
    const double proposed = evaluate_residual(candidate, planes);
    const double predicted = 0.5 * delta.dot(
        damping * diagonal.cwiseProduct(delta) - gradient);
    const double actual = current - proposed;
    const double rho = predicted > 0.0 ? actual / predicted
                                       : -std::numeric_limits<double>::infinity();
    std::cerr << "iter=" << iteration << " residual=" << current
              << " proposed=" << proposed << " damping=" << damping
              << " rho=" << rho << " max_rot=" << max_rotation
              << " max_trans=" << max_translation << "\n";
    if (actual > 0.0 && predicted > 0.0) {
      poses.swap(candidate);
      ++report.accepted;
      report.max_rotation_update =
          std::max(report.max_rotation_update, max_rotation);
      report.max_translation_update =
          std::max(report.max_translation_update, max_translation);
      double factor = 1.0 - std::pow(2.0 * rho - 1.0, 3.0);
      damping *= std::max(1.0 / 3.0, factor);
      nu = 2.0;
      recompute = true;
      if (std::abs(actual) / std::max(std::abs(current), 1e-12) < 1e-6) {
        current = proposed;
        break;
      }
      current = proposed;
    } else {
      damping *= nu;
      nu *= 2.0;
      recompute = false;
    }
  }
  report.final_residual = evaluate_residual(poses, planes);
  report.final_damping = damping;
  return report;
}

void write_tum(
    const std::string& path, const std::vector<double>& timestamps,
    const std::vector<IMUST>& poses) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("cannot write " + path);
  stream << std::fixed << std::setprecision(9);
  for (std::size_t i = 0; i < poses.size(); ++i) {
    Eigen::Quaterniond quaternion(poses[i].R);
    quaternion.normalize();
    stream << timestamps[i] << ' ' << poses[i].p.x() << ' ' << poses[i].p.y()
           << ' ' << poses[i].p.z() << ' ' << quaternion.x() << ' '
           << quaternion.y() << ' ' << quaternion.z() << ' ' << quaternion.w()
           << '\n';
  }
}

int main(int argc, char** argv) {
  try {
    std::string input, tum_path, keyframe_dir, output, report_path;
    int iterations = 10;
    double damping = 0.01;
    double root_voxel = 4.0;
    double downsample_leaf = 0.4;
    int octree_layers = 2;
    double eigen_ratio = 1.0 / 16.0;
    double terminal_eigen_ratio = std::numeric_limits<double>::quiet_NaN();
    int layer_point_threshold = 30;
    int minimum_points = 15;
    bool dense_audit = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      auto value = [&]() -> std::string {
        if (++i >= argc) throw std::runtime_error("missing argument value");
        return argv[i];
      };
      if (argument == "--input") input = value();
      else if (argument == "--tum") tum_path = value();
      else if (argument == "--keyframe-dir") keyframe_dir = value();
      else if (argument == "--output-tum") output = value();
      else if (argument == "--report") report_path = value();
      else if (argument == "--iterations") iterations = std::stoi(value());
      else if (argument == "--damping") damping = std::stod(value());
      else if (argument == "--root-voxel") root_voxel = std::stod(value());
      else if (argument == "--downsample-leaf") downsample_leaf = std::stod(value());
      else if (argument == "--octree-layers") octree_layers = std::stoi(value());
      else if (argument == "--eigen-ratio") eigen_ratio = std::stod(value());
      else if (argument == "--terminal-eigen-ratio") {
        terminal_eigen_ratio = std::stod(value());
      }
      else if (argument == "--layer-point-threshold") {
        layer_point_threshold = std::stoi(value());
      } else if (argument == "--minimum-points") {
        minimum_points = std::stoi(value());
      }
      else if (argument == "--dense-audit") dense_audit = true;
      else throw std::runtime_error("unknown argument: " + argument);
    }
    const bool fixed_mode = !input.empty();
    const bool official_association_mode = !tum_path.empty() && !keyframe_dir.empty();
    if (fixed_mode == official_association_mode || output.empty() || report_path.empty()) {
      throw std::runtime_error(
          "choose either --input or (--tum and --keyframe-dir), plus outputs");
    }
    if (!std::isfinite(terminal_eigen_ratio)) {
      terminal_eigen_ratio = eigen_ratio;
    }
    Dataset data = fixed_mode
        ? load_dataset(input)
        : load_official_associations(
              tum_path, keyframe_dir, root_voxel, downsample_leaf,
              octree_layers, eigen_ratio, terminal_eigen_ratio,
              layer_point_threshold, minimum_points);
    if (dense_audit) verify_unmodified_dense(data, 1e-7);
    const auto started = std::chrono::steady_clock::now();
    OptimizationReport report = optimize(
        data.poses, data.planes, iterations, damping);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    write_tum(output, data.timestamps, data.poses);
    std::ofstream json(report_path);
    json << std::setprecision(12)
         << "{\n"
         << "  \"solver\": \"official_balm2_exact_sparse_adapter\",\n"
         << "  \"official_commit\": \"5dc1bf927fcb65ef17f0e687f553c234d6b17365\",\n"
         << "  \"association\": \""
         << (fixed_mode ? "ghostloop_frozen" : "official_adaptive_voxel")
         << "\",\n"
         << "  \"root_voxel_size\": " << root_voxel << ",\n"
         << "  \"downsample_leaf\": " << downsample_leaf << ",\n"
         << "  \"octree_layers\": " << octree_layers << ",\n"
         << "  \"eigen_ratio\": " << eigen_ratio << ",\n"
         << "  \"terminal_eigen_ratio\": " << terminal_eigen_ratio << ",\n"
         << "  \"layer_point_threshold\": " << layer_point_threshold << ",\n"
         << "  \"minimum_points\": " << minimum_points << ",\n"
         << "  \"pose_count\": " << data.poses.size() << ",\n"
         << "  \"plane_count\": " << data.planes.size() << ",\n"
         << "  \"attempted_iterations\": " << report.attempted << ",\n"
         << "  \"accepted_iterations\": " << report.accepted << ",\n"
         << "  \"initial_residual\": " << report.initial_residual << ",\n"
         << "  \"final_residual\": " << report.final_residual << ",\n"
         << "  \"final_damping\": " << report.final_damping << ",\n"
         << "  \"max_rotation_update_rad\": " << report.max_rotation_update << ",\n"
         << "  \"max_translation_update_m\": " << report.max_translation_update << ",\n"
         << "  \"elapsed_sec\": " << elapsed << ",\n"
         << "  \"dense_official_audit\": " << (dense_audit ? "true" : "false") << "\n"
         << "}\n";
    std::cout << "residual " << report.initial_residual << " -> "
              << report.final_residual << ", accepted " << report.accepted
              << "/" << report.attempted << ", elapsed " << elapsed << " s\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
