#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
typedef pcl::PointXYZ PointT;

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

void selectwithindistance(
	const pcl::PointCloud<PointT>::Ptr &cloud,
	const pcl::ModelCoefficients &model,
	const double threshold,
	std::vector<int> &index)
{
	std::vector<int>().swap(index);
	double sqr_threshold = threshold * threshold;
	Eigen::Vector4f line_pt(model.values[0], model.values[1], model.values[2], 0);
	Eigen::Vector4f line_dir(model.values[3], model.values[4], model.values[5], 0);
	line_dir.normalize();
	for (int i = 0; i < cloud->size(); i++)
	{
		if (cloud->points[i].x < model.values[0] - 30 * threshold || cloud->points[i].x > model.values[0] + 30 * threshold || cloud->points[i].y < model.values[1] - 30 * threshold || cloud->points[i].y > model.values[1] + 30 * threshold)
			continue;
		double sqr_distance = (line_pt - cloud->points[i].getVector4fMap()).cross3(line_dir).squaredNorm();
		if (sqr_distance < sqr_threshold)
			index.push_back(i);
	}
}

auto loadStemFromTXT(const string &fntxt)
{
	vector<vector<float>> stem;
	load(fntxt, stem);
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->reserve(stem.size());
	size_t idx = stem[0].size() - 1;
	for (size_t i = 0; i < stem.size(); i++)
	{
		if (stem[i][idx] == 0)
			continue;
		else
			cloud->push_back(PointT(stem[i][0], stem[i][1], stem[i][2]));
	}
	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(10);
	sor.setStddevMulThresh(2.0);
	sor.filter(*cloud_filtered);
	return cloud_filtered;
}

auto vector2cloud(const vector<vector<float>> &clouddata)
{
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	cloud->height = 1;
	cloud->is_dense = true;
	cloud->resize(clouddata.size());
	for (size_t i = 0; i < clouddata.size(); i++)
	{
		(*cloud)[i].x = clouddata[i][0];
		(*cloud)[i].y = clouddata[i][1];
		(*cloud)[i].z = clouddata[i][2];
	}
	return cloud;
}

auto euclideanCluster(const pcl::PointCloud<PointT>::Ptr &cloud)
{
	pcl::PointCloud<PointT>::Ptr cloud2D(new pcl::PointCloud<PointT>);
	cloud2D->height = 1;
	cloud2D->is_dense = true;
	cloud2D->resize(cloud->size());
	for (size_t i = 0; i < cloud->size(); i++)
	{
		(*cloud2D)[i].x = (*cloud)[i].x;
		(*cloud2D)[i].y = (*cloud)[i].y;
		(*cloud2D)[i].z = 0;
	}
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(cloud2D);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.3);	
	ec.setMaxClusterSize(1000000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud2D);
	ec.extract(cluster_indices);
	return cluster_indices;
}
void exit_with_help()
{
	
	PCL_ERROR( "Usage:   stemExtract [cloud_of_all_points] [pointCNN_result] [radius]\n");
	PCL_ERROR( "Example: stemExtract testdata_original.txt pointcnn_test.txt 0.04\n");
	exit(-1);
}

int main(int argc, char **argv)
{
	if (argc != 4)
		exit_with_help();
	string file_input_data = argv[1];
	string file_input_stem = argv[2];
	float stemRadius = atof(argv[3]);

	// stem grows vertically, hence the angle between the stem and z-axis should be less than 10 degree
	float cosAngleThreshold = cosf(10.0 / 180.0 * M_PI);

	string file_output_stem = file_input_stem;
	string file_output_coeff = file_input_stem;
	file_output_stem.replace(file_output_stem.length() - 4, 4, "_refined.txt");
	file_output_coeff.replace(file_output_coeff.length() - 4, 4, "_stem_coeffs.txt");

	auto start = chrono::system_clock::now();
	auto cloud = loadStemFromTXT(file_input_stem);
	auto cluster_indices = euclideanCluster(cloud);

	vector<vector<float>> data_total;
	load(file_input_data, data_total);
	auto cloud_total = vector2cloud(data_total);
	auto stop = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	printf(" Complete read point cloud for %d points, Used time:%ld ms\n", data_total.size(), duration.count());

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	std::vector<pcl::ModelCoefficients> sac_coeffs;
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_STICK);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000);
	seg.setDistanceThreshold(stemRadius);
	seg.setInputCloud(cloud);

	for (int i = 0; i < cluster_indices.size(); i++)
	{
		pcl::IndicesPtr sac_index(new vector<int>());
		sac_index->assign(cluster_indices[i].indices.begin(), cluster_indices[i].indices.end());

		seg.setIndices(sac_index);
		seg.segment(*inliers, *coefficients);

		// stem grows vertically, hence the angle between the stem and z-axis should be less than 10 degree		
		if(coefficients->values[5]>= cosAngleThreshold)
			sac_coeffs.push_back(*coefficients);
	}
	stop = chrono::system_clock::now();
	duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	printf(" Complete RACSAC segment for %d clusters\n", sac_coeffs.size());

	ofstream ofsCoeff(file_output_coeff);
	std::vector<int> label_total(cloud_total->size(), 0);
	for (int i = 0; i < sac_coeffs.size(); i++)
	{
		ofsCoeff << sac_coeffs[i].values[0] << " "
				 << sac_coeffs[i].values[1] << " "
				 << sac_coeffs[i].values[2] << " "
				 << sac_coeffs[i].values[3] << " "
				 << sac_coeffs[i].values[4] << " "
				 << sac_coeffs[i].values[5] << endl;
		std::vector<int> index;
		selectwithindistance(cloud_total, sac_coeffs[i], stemRadius, index);
		for (int j = 0; j < index.size(); j++)
		{
			label_total[index[j]] = i + 1;
		}
	}
	ofsCoeff.close();

	ofstream ofsCluster(file_output_stem);
	for (int i = 0; i < data_total.size(); i++)
	{
		for (int j = 0; j < data_total[i].size(); j++)
			ofsCluster << data_total[i][j] << " ";
		ofsCluster << label_total[i] << endl;
	}

	ofsCluster.close();
	stop = chrono::system_clock::now();
	duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	printf(" Stem extration Completed. ");
	printf(" Total time used:%ld ms\n", duration.count());
	return (0);
}