//Nontapat Pintira   6088118
//Thanakorn Pasangthien 6088109
//Arada Puengmongkolchaikit 6088133
//Dujnapa Tanundetc 6088105
//Vipawan Jarukitpipat 6088044
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void cvt16UC128UC1(Mat &m_matrix, Mat &dest) {
    Mat out;
    normalize(m_matrix, out, 255, 0, NORM_MINMAX);
    convertScaleAbs(out, dest);
}

int main() {
    
    Mat img = imread("/Users/thanakornpasangthien/Desktop/multimedia/Worapan_Project/Project_MM/Project_MM/test.jpg", IMREAD_COLOR);        // Read in the image. Change the file name.
    
    namedWindow("Actual", WINDOW_AUTOSIZE);            // Window for showing the image
    namedWindow("M_matrix", WINDOW_AUTOSIZE);        // Window for showing the M matrix
    namedWindow("Best seam", WINDOW_AUTOSIZE);        // Window for showing the best seam
    
    imshow("Actual", img);
    
    int c = cvWaitKey(0);
    Size sz = img.size();
    int height = sz.height;
    int width = sz.width;
    
    if (img.empty()) {                                // Return -1 if the image is not found
        return -1;
    }
    
    // ESC = 27, a = 97, d = 100, s = 115, w = 119
    while (c != 27) {
        
        // Looping till get the command 'a', 'd', 'w', 's'
        while (c != 97 && c != 100 && c!= 115 && c != 119 && c != 27) {
            c = cvWaitKey(0);
        }
        
        vector<int> bestSeam;
        
        if (c == 97 || c == 100) {
            
            Mat paddedImage;
            Mat grayImage;
            
            cvtColor(img, grayImage, COLOR_BGR2GRAY);                    // Convert the image to grayscale
            Mat M_matrix(Size(width, height), CV_16UC1, Scalar(0));        // M matrix
            Mat K_matrix(Size(width, height), CV_8UC1, Scalar(0));        // K matrix
            
            copyMakeBorder(grayImage, paddedImage, 1, 1, 1, 1, BORDER_REPLICATE);    // Padding with duplicate using openCV function
            
            int r_M, c_M;        // Row and column in M matrix
            int cL, cU, cR;
            int mL, mU, mR;
            
            // Start from 1 - i-i because the image is padded
            for (int i = 1; i < paddedImage.rows - 1; i++) {
                for (int j = 1; j < paddedImage.cols - 1; j++) {
                    
                    r_M = i - 1;        // Convert i and j to the coordinate in M matrix
                    c_M = j - 1;
                    
                    // Calculate forward energy
                    cL = abs((int)paddedImage.at<uchar>(i, j + 1) - (int)paddedImage.at<uchar>(i, j - 1)) + abs((int)paddedImage.at<uchar>(i - 1, j) - (int)paddedImage.at<uchar>(i, j - 1));
                    cU = abs((int)paddedImage.at<uchar>(i, j + 1) - (int)paddedImage.at<uchar>(i, j - 1));
                    cR = abs((int)paddedImage.at<uchar>(i, j + 1) - (int)paddedImage.at<uchar>(i, j - 1)) + abs((int)paddedImage.at<uchar>(i - 1, j) - (int)paddedImage.at<uchar>(i, j + 1));
                    
                    // Finding Left, Upper and Right M
                    if (c_M == 0 || r_M == 0) {
                        mL = cL;        // Left most, Top row
                    }
                    else {
                        mL = ((int)M_matrix.at<ushort>(r_M - 1, c_M - 1) + cL);
                    }
                    if (r_M == 0) {
                        mU = cU;        // If the image is at the top row
                    }
                    else {
                        mU = ((int)M_matrix.at<ushort>(r_M - 1, c_M) + cU);
                    }
                    if (c_M >= width - 1 || r_M == 0) {
                        mR = cR;        // Right most, Top row
                    }
                    else {
                        mR = ((int)M_matrix.at<ushort>(r_M - 1, c_M + 1) + cR);
                    }
                    
                    // Assign the min M into M matrix
                    M_matrix.at<ushort>(r_M, c_M) = (ushort) min(mL, min(mU, mR));
                    
                    // Find offset and create K matrix
                    if (r_M == 0) {
                        K_matrix.at<uchar>(r_M, c_M) = 0;                        // Top row
                    }
                    else {
                        if ((c_M != 0) && (mL == min(mL, min(mU, mR)))) {
                            K_matrix.at<uchar>(r_M, c_M) = 1;                    // Not left most, Normal
                        }
                        else if (mU == min(mL, min(mU, mR))) {
                            K_matrix.at<uchar>(r_M, c_M) = 2;                    // Normal
                        }
                        else {
                            if (c_M >= width - 1) {
                                K_matrix.at<uchar>(r_M, c_M) = 2;                // Right most
                            }
                            else {
                                K_matrix.at<uchar>(r_M, c_M) = 3;                // Normal
                            }
                        }
                    }
                }
            }
            
            cvWaitKey(1);
            Mat m_show;
            cvt16UC128UC1(M_matrix, m_show);        // Normalize and show the image
            imshow("M_matrix", m_show);
            
            // Matrix for storing the seam
            Mat seam_matrix(Size(width, height), CV_8UC1, Scalar(0));
            
            // Finding the least value in the bottom row of M matrix
            int seam_column = 0;
            for (int i = 0; i < width; i++) {
                if ((int)M_matrix.at<ushort>(height - 1, i) < (int)M_matrix.at<ushort>(height - 1, seam_column)) {
                    seam_column = i;                // Least cumulative energy
                }
            }
            
            seam_matrix.at<uchar>(height - 1, seam_column) = 255;        // Mark the white color on the matrix to indicate the seam
            bestSeam.push_back(seam_column);                            // push_back is like arrayList.add()
            
            // Iterate to find the best seam
            int seam_column_new;
            for (int i = height - 1; i > 0; i--) {
                seam_column_new = seam_column + K_matrix.at<uchar>(i, seam_column) - 2;        // New index is equal to the previous index + offset
                seam_matrix.at<uchar>(i - 1, seam_column_new) = 255;                        // Mark a white pixel
                bestSeam.push_back(seam_column_new);                                        // Add to the best seam
                seam_column = seam_column_new;                                                // Assign new value
            }
            
            cvWaitKey(1);
            imshow("Best seam", seam_matrix);        // Show the image
        }
        
        if (c == 115 || c == 119) {
            
            Mat paddedImage;
            Mat grayImage;
            
            cvtColor(img, grayImage, COLOR_BGR2GRAY);                    // Convert the image to grayscale
            Mat M_matrix(Size(width, height), CV_16UC1, Scalar(0));        // M matrix
            Mat K_matrix(Size(width, height), CV_8UC1, Scalar(0));        // K matrix
            
            /*Perform padding with duplicate using openCV function*/
            copyMakeBorder(grayImage, paddedImage, 1, 1, 1, 1, BORDER_REPLICATE);        // Perform padding
            
            int r_M, c_M;            // Coordinate in M matrix
            int cL, cU, cR;
            int mL, mU, mR;
            
            /*Start from 1 because the image is padded*/
            for (int i = 1; i < paddedImage.cols - 1; i++) {
                for (int j = paddedImage.rows - 2; j > 0; j--) {
                    r_M = j - 1;             // Change i and j to coordinate on M matrix
                    c_M = i - 1;
                    
                    /*Calculate forward energy*/
                    cL = abs((int)paddedImage.at<uchar>(j - 1, i) - (int)paddedImage.at<uchar>(j + 1, i)) + abs((int)paddedImage.at<uchar>(j, i - 1) - (int)paddedImage.at<uchar>(j + 1, i));
                    cU = abs((int)paddedImage.at<uchar>(j - 1, i) - (int)paddedImage.at<uchar>(j + 1, i));
                    cR = abs((int)paddedImage.at<uchar>(j - 1, i) - (int)paddedImage.at<uchar>(j + 1, i)) + abs((int)paddedImage.at<uchar>(j, i - 1) - (int)paddedImage.at<uchar>(j + 1, i));
                    
                    /*Finding Left M, Upper M, and Right M*/
                    if (c_M == 0 || r_M >= height - 1) {
                        mL = cL;                                                    // Left most,top most
                    }
                    else {
                        mL = ((int)M_matrix.at<ushort>(r_M + 1, c_M - 1) + cL);
                    }
                    if (c_M == 0) {
                        mU = cU;                                                    // Top most
                    }
                    else {
                        mU = ((int)M_matrix.at<ushort>(r_M, c_M - 1) + cU);
                    }
                    if (c_M == 0 || r_M == 0) {
                        mR = cR;                                                    // Right most, top most
                    }
                    else {
                        mR = ((int)M_matrix.at<ushort>(r_M - 1, c_M - 1) + cR);
                    }
                    
                    M_matrix.at<ushort>(r_M, c_M) = (ushort)min(mL, min(mU, mR));    // Find min M
                    
                    /*Find offset and assign to K matrix*/
                    if (c_M == 0) {                                                    // Top most
                        K_matrix.at<uchar>(r_M, c_M) = 0;
                    }
                    else {
                        if ((r_M != height - 1) && (mL == min(mL, min(mU, mR)))) {    // Normal
                            K_matrix.at<uchar>(r_M, c_M) = 3;
                        }
                        else if (mU == min(mL, min(mU, mR))) {
                            K_matrix.at<uchar>(r_M, c_M) = 2;
                        }
                        else {
                            if (r_M == 0) {                                            // Right most
                                K_matrix.at<uchar>(r_M, c_M) = 2;
                            }
                            else {
                                K_matrix.at<uchar>(r_M, c_M) = 1;
                            }
                        }
                    }
                }
            }
            
            cvWaitKey(1);
            Mat m_show;
            cvt16UC128UC1(M_matrix, m_show);
            imshow("M_matrix", m_show);
            
            /*Find the best seam in horizontal direction*/
            
            /*Create a matrix for showing the best seam*/
            Mat seam_matrix(Size(width, height), CV_8UC1, Scalar(0));
            
            /*Finding the least value in the bottom row of M matrix*/
            int seam_row = 0;
            for (int i = 0; i < height; i++) {
                if ((int)M_matrix.at<ushort>(i, width - 1) < (int)M_matrix.at<ushort>(seam_row, width - 1)) {            // Finding the least M
                    seam_row = i;
                }
            }
            
            /*Building the best seam and insert it into the matrix*/
            seam_matrix.at<uchar>(seam_row, width - 1) = 255;
            bestSeam.push_back(seam_row);
            int seam_row_new;
            for (int i = width - 1; i > 0; i--) {
                seam_row_new = seam_row + K_matrix.at<uchar>(seam_row, i) - 2;
                seam_matrix.at<uchar>(seam_row_new, i - 1) = 255;
                bestSeam.push_back(seam_row_new);
                seam_row = seam_row_new;
            }
            
            cvWaitKey(1);
            imshow("Best seam", seam_matrix);
        }
        
        if (c == 97) {
            // Reduce width or delete seam vertically
            
            int rowsize = img.rows;
            int colsize = img.cols;
            
            Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));    // Dummy. Because size matters.
            
            for (int i = 0; i < rowsize; i++) {                // Iterate row by row
                
                // segment the image into 2 parts. To the left of the seam and to the right of the seam.
                Mat new_row;
                Mat left = img.rowRange(i, i + 1).colRange(0, bestSeam[i]);
                Mat right = img.rowRange(i, i + 1).colRange(bestSeam[i] + 1, colsize);
                
                // Merge the two segments
                if (!left.empty() && !right.empty()) {
                    hconcat(left, right, new_row);            // Concat horizontally
                    hconcat(new_row, dummy, new_row);
                }
                else {
                    if (left.empty()) {                        // If the left segment is empty, only use the right one
                        hconcat(right, dummy, new_row);
                    }
                    else if (right.empty()) {                // If the right segment is empty, only use the left one
                        hconcat(left, dummy, new_row);
                    }
                }
                new_row.copyTo(img.row(i));                    // Copy the merged row to the image
            }
            
            --width;                                        // Decrement width to fit the new image
            Mat img_new = img.colRange(0, colsize - 1);        // Copy the image into the new matrix
            bestSeam.clear();                                // Reset the seam vector
            
            // Show the resized image
            imshow("Actual", img_new);
            // Clone img_new into img for the next loop processing
            img.release();
            img = img_new.clone();
            img_new.release();
        }
        if (c == 100) {
            // Increase width or insert seam vertically
            int rowsize = img.rows;
            int colsize = img.cols;
            
            int new_width = width + 1;
            Mat img_new(height, new_width, CV_8UC3, Scalar(0, 0, 0));
            
            for (int i = 0; i < rowsize; i++) {
                
                // segment the image into 2 parts. To the left of the seam and to the right of the seam.
                Mat new_row;
                Mat left = img.rowRange(i, i + 1).colRange(0, bestSeam[i]);
                Mat right = img.rowRange(i, i + 1).colRange(bestSeam[i], colsize);
                Mat addseam = img.rowRange(i, i+1).colRange(bestSeam[i], bestSeam[i] +1);
                // Merge the two segments
                if (!left.empty() && !right.empty()) {
                    hconcat(left, addseam, new_row);        // Copy the best seam and concat it with both left and right
                    hconcat(new_row, right, new_row);
                }
                else {
                    if (left.empty()) {
                        hconcat(addseam, right, new_row);
                    }
                    else if (right.empty()) {
                        hconcat(left,addseam, new_row);
                    }
                }
                //insert new row into img_new
                new_row.copyTo(img_new.row(i));
            }
            
            ++width;                                        // Increament the width
            bestSeam.clear();
            
            imshow("Actual", img_new);
            img.release();
            img = img_new.clone();
            img_new.release();
        }
        if (c == 115) {
            // Reduce height or delete seam horizontally
            int rowsize = img.rows;
            int colsize = img.cols;
            
            //Mat img_new(height, --width, CV_8UC3, Scalar(0, 0, 0));
            Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
            
            for (int i = 0; i < colsize; i++) {
                
                // segment the image into 2 parts. To the left of the seam and to the right of the seam.
                Mat new_col;
                Mat upper = img.colRange(i, i+1).rowRange(0, bestSeam[i]);
                Mat lower = img.colRange(i, i+1).rowRange(bestSeam[i] + 1, rowsize);
                
                // Merge the two segments
                if (!upper.empty() && !lower.empty()) {
                    vconcat(upper, lower, new_col);                            // Concat vertically
                    vconcat(new_col, dummy, new_col);
                }
                else {
                    if (upper.empty()) {
                        vconcat(lower, dummy, new_col);
                    }
                    else if (lower.empty()) {
                        vconcat(upper, dummy, new_col);
                    }
                }
                new_col.copyTo(img.col(i));
            }
            
            --height;
            Mat img_new = img.rowRange(0, rowsize - 1);
            bestSeam.clear();
            imshow("Actual", img_new);
            img.release();
            img = img_new.clone();
            img_new.release();
        }
        if (c == 119) {
            // Increase height or insert seam horizontally
            int rowsize = img.rows;
            int colsize = img.cols;
            
            int new_height = height + 1;
            Mat img_new(new_height, width, CV_8UC3, Scalar(0, 0, 0));
            
            for (int i = 0; i < colsize; i++) {
                
                // segment the image into 2 parts. To the left of the seam and to the right of the seam.
                Mat new_col;
                Mat upper = img.colRange(i, i+1).rowRange(0, bestSeam[i]);
                Mat lower = img.colRange(i, i + 1).rowRange(bestSeam[i], rowsize);
                Mat addseam = img.colRange(i, i+1).rowRange(bestSeam[i], bestSeam[i] +1);
                // Merge the two segments
                if (!upper.empty() && !lower.empty()) {
                    vconcat(upper, addseam, new_col);
                    vconcat(new_col, lower, new_col);
                }
                else {
                    if (upper.empty()) {
                        vconcat(addseam, lower, new_col);
                    }
                    else if (lower.empty()) {
                        vconcat(upper, addseam, new_col);
                    }
                }
                new_col.copyTo(img_new.col(i));
            }
            
            ++height;
            
            imshow("Actual", img_new);
            img.release();
            img = img_new.clone();
            img_new.release();
        }
        if (c == 27) break;
        c = cvWaitKey(0);
    }
    return 0;
}

