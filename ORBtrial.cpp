#include"opencv2/opencv.hpp"
#include<highgui.h>
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
/* ./ORBtrial.out img_obj.jpg img_scene.jpg 1.3 3 0 31 9 80 ->default call*/
using namespace cv;
using namespace std;

ofstream f1;
int main(int argc, char **argv)
{
    if(argc<3)
    {
       printf("\n Enter image names!");
       }
   
    Mat img_obj=imread(argv[1],0);
    Mat img_scene=imread(argv[2],0);
    f1.open("OrbClassTimes.txt");
   // ORB orb;
    
  /*  ORB::CommonParams.scale_factor=atof(argv[3]);
    ORB::CommonParams.n_levels=atoi(argv[4]);
    ORB::CommonParams.first_level=atoi(argv[5]);
    ORB::CommonParams.edge_threshold=atoi(argv[6]);;
    */
      /*  if(argc!=8)
        {
            printf("\n Missing Parameters!");
        }*/
    
        float scale_factor=atof(argv[3]);
        int n_levels=atoi(argv[4]);
        int first_level=atoi(argv[5]);
        int edge_threshold=atoi(argv[6]);
        ORB::CommonParams detector_params(scale_factor,n_levels,edge_threshold,first_level); 
        int num_points=atoi(argv[8]);
        ORB orb(num_points,detector_params);
        int ransacThresh=atoi(argv[7]);
        
   /* else
    {
        float scale_factor=1.2;
        int n_levels=3;
        int first_level=0;
        int edge_threshold=31;
        ORB::CommonParams detector_params(scale_factor,n_levels,edge_threshold,first_level); 
        ORB orb(500,detector_params);
        }*/
    //cout<<"scale_factor=";
    //ORB feature detector
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
  //  Mat img_mask=img_obj;
    double t=getTickCount();
    orb(img_obj,img_obj,keypoints1);
    //cout<<"keypoints"<<keypoints1
    orb(img_scene,img_scene,keypoints2);
    t=(getTickCount()-t)/getTickFrequency();
    f1<<"Feature detector:"<<t;
    //ORB feature extractor
    
    int size=orb.descriptorSize();
    Mat obj_des;
    Mat sce_des;
    double t1=getTickCount();
    orb(img_obj,img_obj,keypoints1,obj_des,false);
    orb(img_scene,img_scene,keypoints2,sce_des,false);
    t1=(getTickCount()-t1)/getTickFrequency();
    f1<<"\nFeature Extractor:"<<t1;
    //Descriptor Matching
    
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-HammingLUT");
    
    vector<DMatch> matches;
    double t2=getTickCount();
    matcher->match(obj_des,sce_des,matches);
    t2=(getTickCount()-t2)/getTickFrequency();
    f1<<"\n Descriptor matcher:"<<t2;
    //TODO:Cross check filter can be implemented to better the matches

    //Draw matches
    
    Mat img_out;
    
    drawMatches(img_obj,keypoints1,img_scene,keypoints2,matches,img_out,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    vector<Point2f>obj_points;
    vector<Point2f>scene_points;
    
    vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
    for( size_t j = 0; j < matches.size(); j++ )
    {
        queryIdxs[j] = matches[j].queryIdx;
        trainIdxs[j] = matches[j].trainIdx;
    }
    
   // KeyPoint::convert(keypoints1, obj_points, queryIdxs);
   // KeyPoint::convert(keypoints2, scene_points, trainIdxs);
   
    
     for( int i = 0; i < matches.size(); i++ )
     {
       //-- Get the keypoints from the good matches
       obj_points.push_back( keypoints1[ matches[i].queryIdx ].pt );
       scene_points.push_back( keypoints2[ matches[i].trainIdx ].pt ); 
     }
    
    Mat H=findHomography(obj_points,scene_points,CV_RANSAC, ransacThresh);
    
    vector<Point2f>obj_corners(4);
    vector<Point2f>scene_corners(4);
    Point pt1(0,0),pt2(img_obj.cols,0),pt3(img_obj.cols,img_obj.rows),pt4(0,img_obj.rows);
    
    obj_corners[0]= pt1;
    obj_corners[1]=pt2;
    obj_corners[3]=pt4;   
    obj_corners[2]=pt3;
    
    
    perspectiveTransform( obj_corners, scene_corners, H);
    
    cout<<"\n"<<scene_corners[0]<<endl<<scene_corners[1]<<endl<<scene_corners[2]<<endl<<scene_corners[3]<<endl;
    cout<<"\n"<<img_scene.rows<<endl<<img_scene.cols<<endl;
    
    line( img_scene, scene_corners[0], scene_corners[1] , Scalar(255, 255, 255), 10 );
     
    line( img_scene, scene_corners[1], scene_corners[2] , Scalar( 255, 255, 255), 10 );
    line( img_scene, scene_corners[2] , scene_corners[3] , Scalar( 255, 255, 255), 10 );
    line( img_scene, scene_corners[3]  ,  scene_corners[0] , Scalar( 255, 255, 255), 10 );
    
    
    f1.close();
    
    imshow("object image",img_obj);
    imshow("scene",img_scene);
    imshow("Final",img_out);
    waitKey(0);
    return 0;
    }
