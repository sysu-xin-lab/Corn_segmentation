#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <set>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

using namespace std;
typedef pcl::PointXYZL PointT;

inline unsigned int computeCov2D(const pcl::PointCloud<PointT> &cloud,
								 const std::vector<int> &indices,
								 Eigen::Matrix3f &covariance_matrix,
								 Eigen::Vector4f &xyz_centroid)
{

	if (cloud.empty() || indices.empty())
		return (0);

	// Initialize to 0
	covariance_matrix.setZero();
	unsigned point_count;
	float ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	float nor_x = 0.0f, nor_y = 0.0f, nor_z = 0.0f;
	if (cloud.is_dense)
	{
		point_count = indices.size();
		for (std::vector<int>::const_iterator iIt = indices.begin(); iIt != indices.end(); ++iIt)
		{
			ave_x += cloud[*iIt].x;
			ave_y += cloud[*iIt].y;
			ave_z += cloud[*iIt].z;
		}
		ave_x /= static_cast<float>(point_count);
		ave_y /= static_cast<float>(point_count);
		ave_z /= static_cast<float>(point_count);
		for (std::vector<int>::const_iterator iIt = indices.begin(); iIt != indices.end(); ++iIt)
		{
			nor_x = cloud[*iIt].x - ave_x;
			nor_y = cloud[*iIt].y - ave_y;
			covariance_matrix(0, 0) += nor_x * nor_x;
			covariance_matrix(1, 1) += nor_y * nor_y;
			covariance_matrix(0, 1) += nor_x * nor_y;
		}
		covariance_matrix(1, 0) = covariance_matrix(0, 1);
		covariance_matrix /= static_cast<float>(point_count);
		covariance_matrix(2, 2) = 1;
	}
	else
	{
		point_count = 0;
		for (std::vector<int>::const_iterator iIt = indices.begin(); iIt != indices.end(); ++iIt)
		{
			if (!isFinite(cloud[*iIt]))
				continue;
			++point_count;
			ave_x += cloud[*iIt].x;
			ave_y += cloud[*iIt].y;
			ave_z += cloud[*iIt].z;
		}
		ave_x /= static_cast<float>(point_count);
		ave_y /= static_cast<float>(point_count);
		ave_z /= static_cast<float>(point_count);
		for (std::vector<int>::const_iterator iIt = indices.begin(); iIt != indices.end(); ++iIt)
		{
			if (!isFinite(cloud[*iIt]))
				continue;
			nor_x = cloud[*iIt].x - ave_x;
			nor_y = cloud[*iIt].y - ave_y;
			covariance_matrix(0, 0) += nor_x * nor_x;
			covariance_matrix(1, 1) += nor_y * nor_y;
			covariance_matrix(0, 1) += nor_x * nor_y;
		}
		covariance_matrix(1, 0) = covariance_matrix(0, 1);
		covariance_matrix /= static_cast<float>(point_count);
		covariance_matrix(2, 2) = 1;
	}
	xyz_centroid[0] = ave_x;
	xyz_centroid[1] = ave_y;
	xyz_centroid[2] = ave_z;
	xyz_centroid[3] = 0;
	return (point_count);
}

void computeDirectionAndCenter(
	pcl::PointCloud<PointT>::Ptr &cloud,
	std::vector<int> &indices,
	Eigen::Vector2f &direction,
	PointT &xy_centroid)
{
	Eigen::Matrix3f covariance_matrix;
	Eigen::Vector4f xyz_centroid;
	computeCov2D(*cloud, indices, covariance_matrix, xyz_centroid);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance_matrix);
	Eigen::MatrixXf evecs = eig.eigenvectors();
	//cout<<evecs<<endl<<endl;
	//cout<<eig.eigenvalues()<<endl<<endl;
	float x = evecs(0, 1);
	float y = evecs(1, 1);
	direction << y / x, (xyz_centroid[1] - y / x * xyz_centroid[0]);
	xy_centroid.x = xyz_centroid[0];
	xy_centroid.y = xyz_centroid[1];
}

int minDistanceToLine(
	pcl::PointCloud<PointT>::Ptr &stemPosition,
	std::vector<int> &indices,
	Eigen::Vector2f &leafDirection)
{
	int index = -1;
	float minDisance = FLT_MAX;
	float k = leafDirection[0];
	float b = leafDirection[1];
	for (auto it = indices.begin(); it != indices.end(); ++it)
	{
		float x = (*stemPosition)[*it].x;
		float y = (*stemPosition)[*it].y;
		float distance = std::abs(k * x - y + b);
		if (distance < minDisance)
		{
			minDisance = distance;
			index = (*stemPosition)[*it].label;
		}
	}
	return index; //
}

float inline compute2DSqureDistance(const PointT &a, const PointT &b)
{
	return pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
}

pcl::PointCloud<PointT>::Ptr compute2DCenter(const pcl::PointCloud<PointT>::Ptr &stem,
											 const std::set<int> &stem_labels)
{
	pcl::PointCloud<PointT>::Ptr stemCenter(new pcl::PointCloud<PointT>);
	int max_label = 0;
	for (auto it = stem_labels.begin(); it != stem_labels.end(); it++)
	{
		max_label = std::max(max_label, *it);
	}
	vector<float> sumx(max_label + 1, 0);
	vector<float> sumy(max_label + 1, 0);
	vector<float> number(max_label + 1, 0);
	for (int i = 0; i < stem->size(); i++)
	{
		int idx = (*stem)[i].label;
		sumx[idx] += (*stem)[i].x;
		sumy[idx] += (*stem)[i].y;
		number[idx] += 1;
	}
	for (int idx = 0; idx < number.size(); idx++)
	{
		if (number[idx] != 0)
		{
			PointT pt;
			pt.x = sumx[idx] / number[idx];
			pt.y = sumy[idx] / number[idx];
			pt.z = 0;
			pt.label = idx;
			stemCenter->push_back(pt);
		}
	}
	return stemCenter;
}

template <typename T>
int load(const string &filename, vector<vector<T>> &vec)
{
	ifstream ifs(filename);
	if (!ifs.is_open())
	{
		printf("invaild file: %s\n", filename.c_str());
		exit_with_help();
	}
	string str;

	while (getline(ifs, str))
	{
		T data;
		stringstream ss(str);
		vector<T> v;
		while (ss >> data)
			v.push_back(data);
		vec.push_back(v);
	}
	ifs.close();
	return 0;
}

void writeTXT(const pcl::PointCloud<PointT>::Ptr &cloud,
			  const string filename)
{
	ofstream ofs(filename);
	for (int i = 0; i < cloud->size(); i++)
	{
		ofs << (*cloud)[i].x << " "
			<< (*cloud)[i].y << " "
			<< (*cloud)[i].z << " "
			<< (*cloud)[i].label << endl;
	}
	ofs.close();
}
void exit_with_help()
{	
	PCL_ERROR( "Usage:   cornExtract [cloud_with_stem_label] [radius] [percentile]\n");
	PCL_ERROR( "Example: cornExtract pointcnn_test_refined.txt 0.03 5\n");
	exit(-1);
}

void ecStemCluster(const pcl::PointCloud<PointT>::Ptr &inputCloud,
				   int stemID,
				   float radius,
				   vector<vector<int>> &result)
{
	pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
	kdtree->setInputCloud(inputCloud);
	vector<vector<int>> nghbrs(inputCloud->size());

	std::vector<int> indices;
	for (int i = 0; i < inputCloud->size(); i++)
	{
		vector<float> dist;
		kdtree->radiusSearch(i, radius, nghbrs[i], dist);
		if (inputCloud->points[i].label == stemID)
			indices.push_back(i);
	}

	deque<bool> processed(nghbrs.size(), false);

	for (int i = 0; i < indices.size(); i++)
	{
		int original_idx = indices[i];
		int sq_idx = 0;

		if (processed[original_idx] || nghbrs[original_idx].size() < 2)
			continue;
		vector<int> seed_queue;
		seed_queue.push_back(original_idx);
		processed[original_idx] = true;
		while (sq_idx < static_cast<int>(seed_queue.size()))
		{
			int idx = seed_queue[sq_idx]; // original_idx
			if (nghbrs[idx].size() < 2)
			{
				sq_idx++;
				continue;
			}
			auto curr_nghbr = nghbrs[idx];
			for (std::size_t j = 0; j < curr_nghbr.size(); ++j)
			{
				if (processed[curr_nghbr[j]]) // Has this point been processed before ?
					continue;
				if (nghbrs[curr_nghbr[j]].size() >= 2)
				{
					seed_queue.push_back(curr_nghbr[j]);
				}
				processed[curr_nghbr[j]] = true;
			}
			sq_idx++;
		}
		result.push_back(seed_queue);
	}
}

bool IsConnect(const pcl::PointCloud<PointT>::Ptr &validateCloud,
			   const pcl::PointCloud<PointT>::Ptr &partCloud,
			   int stemID, float radius)

{
	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	for (int i = 0; i < validateCloud->size(); i++)
		inputCloud->push_back(validateCloud->points[i]);
	for (int i = 0; i < partCloud->size(); i++)
		inputCloud->push_back(partCloud->points[i]);

	vector<vector<int>> result;
	ecStemCluster(inputCloud, stemID, radius, result);
	/*
	ofstream ofs2("D:/Projects/HSH/corn-result3/Cluster_" + to_string(stemID) + ".txt");
	for (int i = 0; i < result.size(); i++)
		for (auto & idx : result[i])
			inputCloud->points[idx].label = i + 1;

	for (auto & pt : inputCloud->points)
		ofs2 << pt.x << " " << pt.y << " " << pt.z << " " << pt.label << endl;
	ofs2.close();
	*/
	for (auto &cc : result)
		for (auto &idx : cc)
			if (idx < validateCloud->size())
				return true;
	return false;
}
bool inline isPointEqual(const PointT &a, const PointT &b)
{
	return (a.x == b.x && a.y == b.y && a.z == b.z);
};

void cutConnectLeaf(const pcl::PointCloud<PointT>::Ptr &stem,
					const set<int> &stem_invovled,
					float radius,
					float densityThreshold,
					pcl::PointCloud<PointT>::Ptr &muiltiLeaf)
{
	int temp_size = *max_element(stem_invovled.begin(), stem_invovled.end());
	vector<PointT> connectPts;
	vector<PointT> stemCenter(temp_size + 1);
	vector<int> count(temp_size + 1, 0);
	for (int i = 0; i < stem->size(); i++)
	{
		auto pt = stem->points[i];
		if (find(stem_invovled.begin(), stem_invovled.end(), pt.label) != stem_invovled.end()) //stem
		{
			stemCenter[pt.label].x += pt.x;
			stemCenter[pt.label].y += pt.y;
			count[pt.label] += 1;
			stemCenter[pt.label].label = pt.label;
		}
	}
	for (int i = 0; i < stemCenter.size(); i++)
	{
		if (count[i] != 0)
		{
			stemCenter[i].x /= float(count[i]);
			stemCenter[i].y /= float(count[i]);
			stemCenter[i].z = 0;
		}
		else
		{
			stemCenter[i].x = FLT_MIN;
			stemCenter[i].y = FLT_MIN;
			stemCenter[i].z = FLT_MIN;
		}
	}

	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	inputCloud->height = 1;
	inputCloud->is_dense = true;
	inputCloud->reserve(muiltiLeaf->size() + stem->size());
	for (int i = 0; i < muiltiLeaf->size(); i++)
		inputCloud->push_back(muiltiLeaf->points[i]);
	for (int i = 0; i < stem->size(); i++)
		inputCloud->push_back(stem->points[i]);

	//descend sort
	sort(stemCenter.begin(), stemCenter.end(), [&](PointT a, PointT b)
		 { return a.y > b.y; });

	int stemIdx = 0;
	while (stemIdx < stem_invovled.size() - 1)
	{
		float dist = abs(stemCenter[stemIdx].y - stemCenter[stemIdx + 1].y) / 2.0 - 0.05;
		float startPos = stemCenter[stemIdx + 1].y + dist;
		float endPos = stemCenter[stemIdx].y - dist;
		pcl::PointCloud<PointT>::Ptr subCloud(new pcl::PointCloud<PointT>);
		pcl::PointCloud<PointT>::Ptr bCluster(new pcl::PointCloud<PointT>);
		pcl::PointCloud<PointT>::Ptr tCluster(new pcl::PointCloud<PointT>);
		for (int i = 0; i < inputCloud->size(); i++)
		{
			for (auto &pt : connectPts)
				if (isPointEqual(pt, inputCloud->points[i]))
					continue;
			if (inputCloud->points[i].y > startPos && inputCloud->points[i].y < endPos)
				subCloud->push_back(inputCloud->points[i]);
			if (inputCloud->points[i].y <= startPos)
				bCluster->push_back(inputCloud->points[i]);
			if (inputCloud->points[i].y >= endPos)
				tCluster->push_back(inputCloud->points[i]);
		}

		pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
		kdtree->setInputCloud(subCloud);
		pcl::EuclideanClusterExtraction<PointT> ec;
		std::vector<pcl::PointIndices> ec_result;
		ec.setInputCloud(subCloud);
		ec.setClusterTolerance(radius); //5cm
		ec.setMinClusterSize(densityThreshold);
		ec.setMaxClusterSize(300000);
		ec.setSearchMethod(kdtree);
		ec.extract(ec_result);

		pcl::PointCloud<PointT>::Ptr validateCloud(new pcl::PointCloud<PointT>);
		for (int i = 0; i < ec_result.size(); i++)
		{
			auto ec_cluster = ec_result[i];
			validateCloud->clear();
			for (auto idx : ec_cluster.indices)
				validateCloud->push_back(subCloud->points[idx]);
			kdtree->setInputCloud(validateCloud);
			std::vector<int> curr_cluster;

			// is connect to bottom
			bool bConnect = IsConnect(validateCloud, bCluster, stemCenter[stemIdx + 1].label, radius);

			// is connect to top
			bool tConnect = IsConnect(validateCloud, tCluster, stemCenter[stemIdx].label, radius);
			//cout << "ec_cluster:" << i + 1 << boolalpha << " ; bConnect:" << bConnect << "; tConnect:" << tConnect << endl;
			if (bConnect && tConnect)
			{
				for (auto pt : validateCloud->points)
					connectPts.push_back(pt);
			}
		}
		stemIdx++;
	}

	// split muiltiLeaf  into splitedCloud(including leaves and stems) and connectPts
	pcl::PointCloud<PointT>::Ptr splitedCloud(new pcl::PointCloud<PointT>);
	for (int i = 0; i < inputCloud->size(); i++)
	{
		bool isConnect = false;
		for (auto &pt : connectPts)
			if (isPointEqual(pt, inputCloud->points[i]))
			{
				isConnect = true;
				break;
			}
		if (!isConnect)
			splitedCloud->push_back(inputCloud->points[i]);
	}

	// assign label for points in splitedCloud
	for (auto it = stem_invovled.begin(); it != stem_invovled.end(); it++)
	{
		vector<vector<int>> result;
		ecStemCluster(splitedCloud, *it, radius, result);
		for (auto &cc : result)
			for (auto &idx : cc)
				splitedCloud->points[idx].label = *it;
	}

	// assign label for connectPts
	int number = stem_invovled.size();
	for (auto it = connectPts.begin(); it != connectPts.end(); it++)
	{
		int label = 0;
		float minDist = FLT_MAX;
		for (int idx = 0; idx < number; idx++)
		{
			float dist = compute2DSqureDistance(stemCenter[idx], *it);
			if (dist < minDist)
			{
				label = stemCenter[idx].label;
				minDist = dist;
			}
		}
		it->label = label;
	}

	//merge the splitedCloud and connectPts
	for (int i = 0; i < connectPts.size(); i++)
		splitedCloud->push_back(connectPts[i]);

	// assign label for muiltiLeaf
	// splitedCloud = muiltiLeaf + stems
	pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
	kdtree->setInputCloud(splitedCloud);
	for (int i = 0; i < muiltiLeaf->size(); i++)
	{
		auto pt = muiltiLeaf->points[i];
		vector<int> knnIdx;
		vector<float> knnDist;
		kdtree->nearestKSearch(pt, 1, knnIdx, knnDist);
		muiltiLeaf->points[i].label = splitedCloud->points[knnIdx[0]].label;
	}
}

int main(int argc, char **argv)
{
	if (argc != 4)
		exit_with_help();
	string file_input = argv[1];
	float radius = atof(argv[2]);
	float pct = 1-atof(argv[3])/100.0f;
	string file_output = file_input;
	file_output.replace(file_output.length() - 4, 4, "_cluster.txt");
	auto start = chrono::system_clock::now();
	vector<vector<float>> data_total;
	load(file_input, data_total);

	pcl::PointCloud<PointT>::Ptr leaf(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr stem(new pcl::PointCloud<PointT>);
	leaf->height = 1;
	leaf->is_dense = true;
	stem->height = 1;
	stem->is_dense = true;

	set<int> stem_labels;
	for (int i = 0; i < data_total.size(); i++)
	{
		int idx = data_total[0].size() - 1; // the last column indicate the label of stem and leaf
		PointT pt;
		pt.x = data_total[i][0];
		pt.y = data_total[i][1];
		pt.z = data_total[i][2];
		pt.label = data_total[i][idx];
		if (pt.label == 0) //leaf
			leaf->push_back(pt);
		else //stem
		{
			stem->push_back(pt);
			stem_labels.insert(pt.label);
		}
	}
	int number_of_stem = stem_labels.size();

	pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
	kdtree->setInputCloud(leaf);

	vector<vector<int>> nghbrs(leaf->size());
	vector<vector<int>> clusters;
	vector<vector<int>> clusters_tbd;
	vector<std::pair<int, int>> density(leaf->size());
	for (int i = 0; i < leaf->size(); i++)
	{
		vector<float> temp;
		kdtree->radiusSearch(i, radius, nghbrs[i], temp);
		density[i].first = i;
		density[i].second = nghbrs[i].size();
	}

	//descend sort according to the density
	std::sort(density.begin(), density.end(), [&](std::pair<int, int> a, std::pair<int, int> b)
			  { return a.second > b.second; });

	// Perform a clustering
	int densityThreshold = density[int(density.size() * pct)].second; //percentile 95%
	cout << "density threshold: " << densityThreshold << endl;
	std::vector<bool> processed(density.size(), false);

	for (int i = 0; i < density.size(); i++)
	{
		int original_idx = density[i].first;
		if (processed[original_idx] || nghbrs[original_idx].size() < densityThreshold)
			continue;
		std::vector<int> seed_queue;
		std::vector<int> curr_cluster;
		int sq_idx = 0;
		seed_queue.push_back(original_idx);
		curr_cluster.push_back(original_idx);
		processed[original_idx] = true;
		while (sq_idx < static_cast<int>(seed_queue.size()))
		{
			// Search for sq_idx
			int idx = seed_queue[sq_idx]; // original_idx
			if (nghbrs[idx].size() < densityThreshold)
			{
				sq_idx++;
				continue;
			}
			// nghbrs of idx
			auto curr_nghbr = nghbrs[idx];
			for (std::size_t j = 0; j < curr_nghbr.size(); ++j)
			{

				if (processed[curr_nghbr[j]]) // Has this point been processed before ?
					continue;
				// if density is large enough, add it to seed
				if (nghbrs[curr_nghbr[j]].size() >= densityThreshold)
				{
					seed_queue.push_back(curr_nghbr[j]);
				}
				// add it to cluster
				curr_cluster.push_back(curr_nghbr[j]);
				processed[curr_nghbr[j]] = true;
			}
			sq_idx++;
		}
		clusters.push_back(curr_cluster);
	}

	auto stop = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << "Initial cluster complete,Used time:" << duration.count() << " ms" << endl;

	cout << "Refining the clusters ...\n";
	// refine the cluster result according to morphological characteristics

	kdtree->setInputCloud(stem);
	for (size_t i = 0; i < clusters.size(); i++)
	{
		set<int> stem_invovled;
		std::vector<int> curr_nghbrs;
		std::vector<float> temp;
		if (clusters[i].size() < densityThreshold)
			continue;
		for (auto cluster_iter = clusters[i].begin(); cluster_iter != clusters[i].end(); cluster_iter++)
		{
			kdtree->radiusSearch((*leaf)[*cluster_iter], radius, curr_nghbrs, temp);
			if (curr_nghbrs.size() > 0)
			{
				for (auto iter = curr_nghbrs.begin(); iter != curr_nghbrs.end(); iter++)
					stem_invovled.insert(stem->points[*iter].label);
			}
		}
		if (stem_invovled.size() == 0)
		{
			clusters_tbd.push_back(clusters[i]);
			for (auto cluster_iter = clusters[i].begin(); cluster_iter != clusters[i].end(); cluster_iter++)
				(*leaf)[*cluster_iter].label = 0;
		}
		if (stem_invovled.size() == 1)
		{
			for (auto cluster_iter = clusters[i].begin(); cluster_iter != clusters[i].end(); cluster_iter++)
				(*leaf)[*cluster_iter].label = *stem_invovled.begin();
		}
		if (stem_invovled.size() >= 2)
		{
			cout << "Cutting " << stem_invovled.size() << " leaves...\n";
			pcl::PointCloud<PointT>::Ptr muiltiLeaf(new pcl::PointCloud<PointT>);
			muiltiLeaf->reserve(clusters[i].size());
			for (auto idx : clusters[i])
				muiltiLeaf->push_back(leaf->points[idx]);
			cutConnectLeaf(stem, stem_invovled, radius, densityThreshold, muiltiLeaf);
			for (int j = 0; j < clusters[i].size(); j++)
				leaf->points[clusters[i][j]].label = muiltiLeaf->points[j].label;
		}
	}
	//assign labels for clusters_tbd
	pcl::PointCloud<PointT>::Ptr corepts(new pcl::PointCloud<PointT>);
	corepts->height = 1;
	corepts->is_dense = true;
	for (size_t i = 0; i < nghbrs.size(); i++)
	{
		if (nghbrs[i].size() >= densityThreshold && leaf->points[i].label != 0)
			corepts->push_back(leaf->points[i]);
	}
	kdtree->setInputCloud(corepts);

	for (size_t i = 0; i < clusters_tbd.size(); i++)
	{
		vector<int> pointIdxKSearch;
		vector<float> pointSquaredDistance;

		float minDist = FLT_MAX;
		int label = 0;
		for (size_t j = 0; j < clusters_tbd[i].size(); j++)
		{
			int idx = clusters_tbd[i][j];
			kdtree->nearestKSearch(leaf->points[idx], 1, pointIdxKSearch, pointSquaredDistance);
			if (minDist > pointSquaredDistance[0])
			{
				minDist = pointSquaredDistance[0];
				label = corepts->points[pointIdxKSearch[0]].label;
			}
		}
		for (size_t j = 0; j < clusters_tbd[i].size(); j++)
		{
			int idx = clusters_tbd[i][j];
			leaf->points[idx].label = label;
		}
	}

	//assign labels for noises
	for (size_t i = 0; i < leaf->size(); i++)
	{
		if (leaf->points[i].label == 0)
		{
			vector<int> pointIdxKSearch;
			vector<float> pointSquaredDistance;
			kdtree->nearestKSearch(leaf->points[i], 1, pointIdxKSearch, pointSquaredDistance);
			int idx = pointIdxKSearch[0];
			leaf->points[i].label = corepts->points[idx].label;
		}
	}

	ofstream ofsCluster(file_output);
	vector<vector<unsigned int>> colors(int(number_of_stem + 1), vector<unsigned int>(3, 0));
	srand(19901990);
	colors[0][0] = 255;
	colors[0][1] = 255;
	colors[0][2] = 255;
	for (auto i_label = 1; i_label < number_of_stem + 1; i_label++)
	{
		colors[i_label][0] = static_cast<unsigned char>(rand() % 256);
		colors[i_label][1] = static_cast<unsigned char>(rand() % 256);
		colors[i_label][2] = static_cast<unsigned char>(rand() % 256);
	}

	for (int i = 0; i < leaf->size(); i++)
	{
		ofsCluster << (*leaf)[i].x << " "
				   << (*leaf)[i].y << " "
				   << (*leaf)[i].z << " "
				   << (int)colors[(*leaf)[i].label][0] << " "
				   << (int)colors[(*leaf)[i].label][1] << " "
				   << (int)colors[(*leaf)[i].label][2] << " "
				   << (*leaf)[i].label << endl;
	}
	for (int i = 0; i < stem->size(); i++)
	{
		ofsCluster << (*stem)[i].x << " "
				   << (*stem)[i].y << " "
				   << (*stem)[i].z << " "
				   << (int)colors[(*stem)[i].label][0] << " "
				   << (int)colors[(*stem)[i].label][1] << " "
				   << (int)colors[(*stem)[i].label][2] << " "
				   << (*stem)[i].label << endl;
	}
	ofsCluster.close();
	stop = chrono::system_clock::now();
	duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << "Refine cluster extraction complete,Used time:" << duration.count() << " ms" << endl;

	return (0);
}
